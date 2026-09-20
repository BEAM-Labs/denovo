"""GPU-accelerated knapsack decoding for precursor mass constrained (PMC) peptide sequencing."""
import cupy as cp
import numpy as np
import torch
import sys

inference_kernel = cp.RawKernel(
r'''
extern "C" __global__
void inference(float* prob, int* ans, float* aa_mass, float* dp, float* dpMass, int* lock,
               float premass, int length, float grid_size, float tol2, int aa_num) {
    const int AA_num = aa_num;
    const float tol = tol2;
    int w = blockIdx.x;
    int dim = blockDim.x;
    int h = threadIdx.x;
    if(w == 0 || h == 0) return;
    float maxMass = 186.079313 * h;
    int maxW = int(maxMass / grid_size);
    if(w > maxW){
        lock[w * dim + h] = 1;
        return;
    }
    if(h == 1){
        for(int i = 1; i < AA_num; i++){
            if(w == int(aa_mass[i] / grid_size)){
                if(dp[w * dim + h] < prob[(h-1) * AA_num + i]){
                    dp[w * dim + h] = prob[(h-1) * AA_num + i];
                    dpMass[w * dim + h] = aa_mass[i];
                    ans[w * (dim * dim) + 1 * dim + 1] = i;
                }
            }
        }
        lock[w * dim + h] = 1;
        return;
    }
    while(!atomicCAS(lock + w * dim + h - 1, 1, 1));
    __threadfence();
    if(dp[w * dim + h - 1] > 0){
        dp[w * dim + h] = dp[w * dim + h - 1] + prob[(h - 1) * AA_num];
        dpMass[w * dim + h] = dpMass[w * dim + h - 1];
        for(int l = 1; l < h + 1; l++){
            ans[w * (dim * dim) + h * dim + l] = ans[w * (dim * dim) + (h - 1) * dim + l];
        }
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
    for(int i = 1; i < AA_num; i++){
        int minw = int((w * grid_size - aa_mass[i]) / grid_size);
        if(minw >= 0){
            while(!atomicCAS(lock + minw * dim + h - 1, 1, 1));
            __threadfence();
            int preid = ans[minw * (dim * dim) + (h-1) * dim + h - 1];
            if(preid != i){
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
        minw = minw + 1;
        if(minw >= 0){
            while(!atomicCAS(lock + minw * dim + h - 1, 1, 1));
            __threadfence();
            int preid = ans[minw * (dim * dim) + (h-1) * dim + h - 1];
            if(preid != i){
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
    lock[w * dim + h] = 1;
    return;
}
''', 'inference')


def knapDecode(prob, preMass, tol, residues):
    """Run GPU knapsack decoding to find the optimal peptide sequence under mass constraint.

    Args:
        prob: Log-probability tensor of shape (1, seq_len, vocab_size).
        preMass: Precursor neutral mass tensor of shape (1,).
        tol: Mass tolerance in Daltons.
        residues: Ordered dict of amino acid masses.

    Returns:
        List of token indices representing the decoded peptide.
    """
    grid_size = 1
    residues = [0.0] + list(residues.values())
    for i, x in enumerate(residues):
        if x < 0:
            residues[i] = 10000.0
    AAmasses = np.array(residues, dtype=np.float32)
    prob = torch.roll(prob[0], 1, dims=-1)
    AAmasses = cp.array(AAmasses, dtype=cp.float32)
    preMass = float(preMass[0].item() - 18.01)
    tol = cp.float32(tol)

    preMass = cp.float32(preMass)
    grid_size = cp.float32(grid_size)
    prob = torch.where(prob < -20.0, -20.0, prob)
    prob = prob - prob.min() + 0.1
    prob = prob.cpu().numpy().astype(np.float32)
    prob = cp.array(prob, dtype=cp.float32)
    word_num = 40
    length = int(preMass / grid_size + 1)
    ans = cp.zeros((length, word_num + 1, word_num + 1), dtype=cp.int32)
    dpProb = cp.zeros((length, word_num + 1), dtype=cp.float32)
    dpMass = cp.zeros((length, word_num + 1), dtype=cp.float32)
    dpLock = cp.zeros((length, word_num + 1), dtype=cp.int32)
    dpLock[0, :] = 1

    aa_num = len(residues)
    inference_kernel(
        (length,), (word_num + 1,),
        (prob, ans, AAmasses, dpProb, dpMass, dpLock, preMass, length, grid_size, tol, aa_num)
    )
    ans = ans.get()
    dpMass = dpMass.get()

    result = ans[length - 1, word_num, 1:]
    result = np.where(result == 0, aa_num - 1, result - 1)

    sys.stdout.flush()
    return result.tolist()
