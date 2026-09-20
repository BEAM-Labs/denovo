"""GPU-accelerated knapsack decoding v2 with transition and start mask matrices."""
import cupy as cp
import numpy as np
import torch
import sys

inference_kernel = cp.RawKernel(
r'''
extern "C" __global__
void inference(float* prob, int* ans, float* aa_mass, float* dp, float* dpMass, int* lock,
               float premass, int length, float grid_size, float tol2, int aa_num,
               float* trans_mask, float* start_mask) {
    const int AA_num = aa_num;
    const float tol = tol2;
    int w = blockIdx.x;
    int dim = blockDim.x;
    int h = threadIdx.x;

    if(w == 0 || h == 0) return;

    // Check if transition is allowed (mask value > large negative number)
    auto is_valid_transition = [&](int prev_aa, int curr_aa) -> bool {
        return trans_mask[prev_aa * AA_num + curr_aa] > -1000.0;
    };

    float maxMass = 186.079313 * h;
    int maxW = int(maxMass / grid_size);
    if(w > maxW){
        lock[w * dim + h] = 1;
        return;
    }

    // First amino acid: apply start mask
    if(h == 1){
        for(int i = 1; i < AA_num; i++){
            if(w == int(aa_mass[i] / grid_size)){
                if(start_mask[i] > -1000.0) {
                    if(dp[w * dim + h] < prob[(h-1) * AA_num + i]){
                        dp[w * dim + h] = prob[(h-1) * AA_num + i];
                        dpMass[w * dim + h] = aa_mass[i];
                        ans[w * (dim * dim) + 1 * dim + 1] = i;
                    }
                }
            }
        }
        lock[w * dim + h] = 1;
        return;
    }

    // Dynamic programming (h > 1)
    while(!atomicCAS(lock + w * dim + h - 1, 1, 1));
    __threadfence();

    // Case A: extend with blank / repeat
    if(dp[w * dim + h - 1] > 0){
        dp[w * dim + h] = dp[w * dim + h - 1] + prob[(h - 1) * AA_num];
        dpMass[w * dim + h] = dpMass[w * dim + h - 1];
        for(int l = 1; l < h + 1; l++){
            ans[w * (dim * dim) + h * dim + l] = ans[w * (dim * dim) + (h - 1) * dim + l];
        }

        // CTC collapse logic
        int preid = ans[w * (dim * dim) + (h - 1) * dim + h - 1];
        if(preid != 0){
            float tempProb = dp[w * dim + h - 1] + prob[(h - 1) * AA_num + preid];
            if(dp[w * dim + h] < tempProb){
                dpMass[w * dim + h] = dpMass[w * dim + h - 1];
                dp[w * dim + h] = tempProb;
                for(int l = 1; l < h; l++){
                    ans[w * (dim * dim) + h * dim + l] = ans[w * (dim * dim) + (h - 1) * dim + l];
                }
                ans[w * (dim * dim) + h * dim + h] = preid;
            }
        }
    }

    // Case B: append new amino acid
    for(int i = 1; i < AA_num; i++){
        int minw = int((w * grid_size - aa_mass[i]) / grid_size);

        for(int k=0; k<2; k++) {
            if (k==1) minw = minw + 1;

            if(minw >= 0){
                while(!atomicCAS(lock + minw * dim + h - 1, 1, 1));
                __threadfence();

                int preid = ans[minw * (dim * dim) + (h-1) * dim + h - 1];

                // Apply transition mask
                if(preid != i && is_valid_transition(preid, i)){
                    float temp = dpMass[minw * dim + h - 1] + aa_mass[i];

                    if(temp >= w * grid_size && temp < (w + 1) * grid_size && w != length-1 ||
                       w == length - 1 && temp >= premass - tol && temp <= premass + tol){

                        float tempProb = dp[minw * dim + h - 1] + prob[(h-1) * AA_num + i];
                        if(tempProb > dp[w * dim + h]){
                            dp[w * dim + h] = tempProb;
                            dpMass[w * dim + h] = temp;
                            for(int l = 1; l < h; l++){
                                ans[w * (dim * dim) + h * dim + l] = ans[minw * (dim * dim) + (h - 1) * dim + l];
                            }
                            ans[w * (dim * dim) + h * dim + h] = i;
                        }
                    }
                }
            }
        }
    }
    lock[w * dim + h] = 1;
    return;
}
''', 'inference')


def knapDecode(prob, preMass, tol, residues, trans_mask, start_mask):
    """Run GPU knapsack decoding with transition and start mask matrices.

    Args:
        prob: Log-probability tensor of shape (1, seq_len, vocab_size).
        preMass: Precursor neutral mass tensor of shape (1,).
        tol: Mass tolerance in Daltons.
        residues: Ordered dict of amino acid masses.
        trans_mask: Transition mask tensor of shape (vocab_size, vocab_size).
        start_mask: Start mask tensor of shape (vocab_size,).

    Returns:
        List of token indices representing the decoded peptide.
    """
    aa_num = len(residues) + 1

    grid_size = 1
    residues_mass = [0.0] + list(residues.values())
    for i, x in enumerate(residues_mass):
        if x < 0:
            residues_mass[i] = 10000.0

    AAmasses = cp.array(np.array(residues_mass, dtype=np.float32), dtype=cp.float32)

    # Process probability tensor (roll to move blank to index 0)
    prob = torch.roll(prob[0], 1, dims=-1)
    prob = torch.where(prob < -10000.0, -10000.0, prob)
    prob = prob - prob.min() + 0.1
    prob = cp.array(prob.cpu().numpy().astype(np.float32), dtype=cp.float32)

    # Process start mask (roll to match prob alignment)
    start_mask = torch.roll(start_mask, 1, dims=0)
    start_mask_cp = cp.array(start_mask.cpu().numpy().astype(np.float32), dtype=cp.float32)

    # Process transition mask (roll both dims to match prob alignment)
    trans_mask = torch.roll(trans_mask, 1, dims=0)
    trans_mask = torch.roll(trans_mask, 1, dims=1)
    trans_mask_cp = cp.array(trans_mask.cpu().numpy().astype(np.float32), dtype=cp.float32)

    # Scalar setup
    preMass_val = float(preMass[0].item() - 18.01)
    preMass_cp = cp.float32(preMass_val)
    tol_cp = cp.float32(tol)
    grid_size_cp = cp.float32(grid_size)

    # Initialize DP matrices
    word_num = 40
    length = int(preMass_val / grid_size + 1)

    ans = cp.zeros((length, word_num + 1, word_num + 1), dtype=cp.int32)
    dpProb = cp.zeros((length, word_num + 1), dtype=cp.float32)
    dpMass = cp.zeros((length, word_num + 1), dtype=cp.float32)
    dpLock = cp.zeros((length, word_num + 1), dtype=cp.int32)
    dpLock[0, :] = 1

    # Launch kernel
    inference_kernel(
        (length,), (word_num + 1,),
        (prob, ans, AAmasses, dpProb, dpMass, dpLock,
         preMass_cp, length, grid_size_cp, tol_cp, aa_num,
         trans_mask_cp, start_mask_cp)
    )

    # Extract result
    ans = ans.get()
    result = ans[length - 1, word_num, 1:]
    result = np.where(result == 0, aa_num - 1, result - 1)

    sys.stdout.flush()
    return result.tolist()
