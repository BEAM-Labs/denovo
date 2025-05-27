import cupy as cp
import numpy as np
import time
import torch
import sys

# inference_kernel = cp.RawKernel(
# r'''
# extern "C" __global__
# void inference(float* prob, int* ans, float* aa_mass, float* dp, float* dpMass ,int* lock,float premass,int length) {
#     const int AA_num = 28;
#     const float tol = 0.1;
#     int w = blockIdx.x;
#     int dim = blockDim.x;
#     int h = threadIdx.x;
#     if(w == 0 || h == 0)return;
#     int maxMass = int(aa_mass[AA_num - 8] * h);
#     if(w > maxMass){
#         lock[w * dim + h] = 1;
#         return;
#     }
#     if(h == 1){
#         for(int i = 1; i < AA_num; i ++){
#             if(w == int(aa_mass[i])){
#                 if(dp[w * dim + h] < prob[(h-1) * AA_num + i]){
#                     dp[w * dim + h] = prob[(h-1) * AA_num + i];
#                     dpMass[w * dim + h] = aa_mass[i];
#                     ans[w * (dim * dim) + 1 * dim + 1] = i;
#                 }
#             }
#         }
#         lock[w * dim + h] = 1;             
#         return;
#     }
#     while(!atomicCAS(lock + w * dim + h - 1, 1, 1));
#     __threadfence();
#     if(dp[w * dim + h - 1] > 0){
#         dp[w * dim + h] = dp[w * dim + h - 1] + prob[(h - 1) * AA_num];
#         dpMass[w * dim + h] = dpMass[w * dim + h - 1];
#         for(int l = 1; l < h + 1; l++){
#             ans[w * (dim * dim) + h * dim + l] = ans[w * (dim * dim) + (h - 1) * dim + l];
#         }
#         int preid = ans[w * (dim * dim) + (h - 1) * dim + h - 1];
#         if(preid != 0){
#             float tempProb = dp[w * dim + h - 1] + prob[(h - 1) * AA_num + preid];
#             if(dp[w * dim + h] < tempProb){
#                 dpMass[w * dim + h] = dpMass[w * dim + h - 1];
#                 dp[w * dim + h] = tempProb;
#                 for(int l = 1; l < h; l++){
#                     ans[w * (dim * dim) + h * dim + l] = ans[w * (dim * dim) + (h - 1) * dim + l];
#                 }
#                 ans[w * (dim * dim) + h * dim + h] = preid;
#             }
#         }
#     }
#     __threadfence();
#     for(int i = 1; i < AA_num; i ++){
#         int minw = int(w - aa_mass[i]);
#         if(minw >= 0){
#             while(!atomicCAS(lock + minw * dim + h - 1, 1, 1));
#             __threadfence();
#             int preid = ans[minw * (dim * dim) + (h-1) * dim + h - 1];
#             if(preid != i){
#                 float temp = dpMass[minw * dim + h - 1] + aa_mass[i];
#                 if(temp >= w && temp < w + 1 && w != length-1 || w == length - 1 && temp >= premass - tol && temp <= premass + tol){
#                     float tempProb = dp[minw * dim + h - 1] + prob[(h-1) * AA_num + i];
#                     if(tempProb > dp[w * dim + h]){
#                         dp[w * dim + h] = tempProb;
#                         dpMass[w * dim + h] = temp;
#                         for(int l = 1; l < h; l++){
#                             ans[w * (dim * dim) + h * dim + l] = ans[minw * (dim * dim) + (h - 1) * dim + l];
#                         }
#                         ans[w * (dim * dim) + h * dim + h] = i;
#                     }
#                 }
#             }
#         }
#         minw = minw + 1;
#         if(minw >= 0){
#             while(!atomicCAS(lock + minw * dim + h - 1, 1, 1));
#             __threadfence();
#             int preid = ans[minw * (dim * dim) + (h-1) * dim + h - 1];
#             if(preid != i){
#                 float temp = dpMass[minw * dim + h - 1] + aa_mass[i];
#                 if(temp >= w && temp < w + 1 && w != length-1 || w == length - 1 && temp >= premass - tol && temp <= premass + tol){
#                     float tempProb = dp[minw * dim + h - 1] + prob[(h-1) * AA_num + i];
#                     if(tempProb > dp[w * dim + h]){
#                         dp[w * dim + h] = tempProb;
#                         dpMass[w * dim + h] = temp;
#                         for(int l = 1; l < h; l++){
#                             ans[w * (dim * dim) + h * dim + l] = ans[minw * (dim * dim) + (h - 1) * dim + l];
#                         }
#                         ans[w * (dim * dim) + h * dim + h] = i;
#                     }
#                 }
#             }
#         }
#     }
#     __threadfence();
#     lock[w * dim + h] = 1;  
#     return;
# }
# ''', 'inference')

inference_kernel = cp.RawKernel(
r'''
extern "C" __global__
void inference(float* prob, int* ans, float* aa_mass, float* dp, float* dpMass ,int* lock,float premass,int length,float grid_size, float tol2) {
    const int AA_num = 28;
    const float tol = tol2;
    int w = blockIdx.x;
    int dim = blockDim.x;
    int h = threadIdx.x;
    if(w == 0 || h == 0)return;
    float maxMass = aa_mass[AA_num - 8] * h;
    int maxW = int(maxMass / grid_size);
    if(w > maxW){
        lock[w * dim + h] = 1;
        return;
    }
    if(h == 1){
        for(int i = 1; i < AA_num; i ++){
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
    for(int i = 1; i < AA_num; i ++){
        int minw = int((w * grid_size - aa_mass[i]) / grid_size);
        if(minw >= 0){
            while(!atomicCAS(lock + minw * dim + h - 1, 1, 1));
            __threadfence();
            int preid = ans[minw * (dim * dim) + (h-1) * dim + h - 1];
            if(preid != i){
                float temp = dpMass[minw * dim + h - 1] + aa_mass[i];
                if(temp >= w * grid_size && temp < (w + 1) * grid_size && w != length-1 || w == length - 1 && temp >= premass - tol && temp <= premass + tol){
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
                if(temp >= w * grid_size && temp < (w + 1) * grid_size && w != length-1 || w == length - 1 && temp >= premass - tol && temp <= premass + tol){
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

def knapDecode(prob, preMass, tol):
    grid_size = 1
    AAmasses = [0.0, 57.021464, 71.037114, 87.032028, 97.052764, 99.068414, 101.04767, 160.030649, 113.084064, 113.084064, 114.042927, 115.026943, 128.058578, 128.094963, 129.042593, 131.040485, 137.058912, 147.068414, 156.101111, 163.063329, 186.079313, 147.0354, 115.026943, 129.042594, 42.010565, 43.005814, 10000.0, 25.980265]
    AAmasses = np.array(AAmasses,dtype=np.float32)
    prob = torch.roll(prob[0],1,dims = -1)
    AAmasses = cp.array(AAmasses,dtype=cp.float32)
    preMass = float(preMass[0].item() - 18.01)
    preMass = cp.float32(preMass)
    grid_size = cp.float32(grid_size)
    tol = cp.float32(tol)
    # print(preMass)
    prob = torch.where(prob < -20.0, -20.0, prob)
    #print(prob)
    #print(prob.min())
    prob = prob - prob.min() + 0.1
    prob = prob.cpu().numpy().astype(np.float32)
    # print(prob)
    prob = cp.array(prob,dtype=cp.float32)
    #print(prob)
    word_num = 40
    length = int(preMass / grid_size + 1)
    ans = cp.zeros((length, word_num + 1, word_num + 1),dtype = cp.int32)
    dpProb = cp.zeros((length,word_num + 1),dtype = cp.float32)
    dpMass = cp.zeros((length,word_num + 1),dtype = cp.float32)
    dpLock = cp.zeros((length,word_num + 1),dtype = cp.int32)
    dpLock[0,:] = 1
    inference_kernel((length,),(word_num + 1,),(prob,ans,AAmasses,dpProb,dpMass,dpLock,preMass,length,grid_size, tol))
    ans = ans.get()
    dpMass = dpMass.get()
    # # print(ans[:,4,1:])
    # print(ans[length-1,word_num,1:])
    #print("desired mass:", preMass)
    #print("decoded mass: ", dpMass[length-1,word_num])
    
    
    # print("time:", end - start)
    result = ans[length-1,word_num,1:]
    result = np.where(result == 0, 27, result - 1)
    #print("result:", result)
    sys.stdout.flush()
    
    return result.tolist()



# cp.random.seed(3407)

# word_num = 40
# aa_num = 28
# prob = cp.random.uniform(-15.0,1.0,size=(word_num,aa_num),dtype=cp.float32)
# print(prob)
# preMass = cp.random.uniform(2800,3000,dtype=cp.float32)
# print(preMass)
# length = int(preMass + 1)
# ans = cp.zeros((length, word_num + 1, word_num + 1),dtype = cp.int32)

# AAmasses = [0.0, 57.021464, 71.037114, 87.032028, 97.052764, 99.068414, 101.04767, 160.030649, 113.084064, 113.084064, 114.042927, 115.026943, 128.058578, 128.094963, 129.042593, 131.040485, 137.058912, 147.068414, 156.101111, 163.063329, 186.079313, 147.0354, 115.026943, 129.042594, 10000.0, 10000.0, 10000.0, 10000.0]
# AAmasses = np.array(AAmasses,dtype=np.float32)
# AAmasses = cp.array(AAmasses,dtype=cp.float32)
# # AAmasses = cp.sort(AAmasses)
# print(AAmasses)

# ans = cp.zeros((length, word_num + 1, word_num + 1),dtype = cp.int32)
# dpProb = cp.zeros((length,word_num + 1),dtype = cp.float32)
# dpMass = cp.zeros((length,word_num + 1),dtype = cp.float32)
# dpLock = cp.zeros((length,word_num + 1),dtype = cp.int32)
# dpLock[0,:] = 1

# preMass = cp.float32(preMass.get())
# start = time.time()
# inference_kernel((length,),(word_num + 1,),(prob,ans,AAmasses,dpProb,dpMass,dpLock,preMass,length))
# end = time.time()
# ans = ans.get()
# dpMass = dpMass.get()
# # print(ans[:,4,1:])
# print(ans[length-1,word_num,1:])
# print(dpMass[length-1,word_num])
# print("time:", end - start)
