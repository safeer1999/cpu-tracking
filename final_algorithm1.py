# Python3 program to check if any 
# Subarray of size K has a given Sum  
  
# Function to check if any Subarray  
# of size K has a given Sum  
def checkSubarraySum(arr, n,  
                     k): 
    # Check for first window  
    curr_sum = 0
    final_sum=0
    start=0
    end=0
    for i in range(0, k):  
        curr_sum += arr[i]  
    if (curr_sum > final_sum):  
        final_sum=curr_sum
        start=0
        end=k
  
    # Consider remaining blocks  
    # ending with j  
    for j in range(k, n):  
        curr_sum = (curr_sum + arr[j] - arr[j - k])  
        if (curr_sum > final_sum) : 
            final_sum=curr_sum
            start=j-k+1
            end=j
    print(start)
    print(end)
    return final_sum
  
# Driver code  
arr = [1,2,3,2,1,66,5,4,3,2,4,2,1,3,4,56,7,65,5,6,4,6]
k = len(arr)//3
  
n = len(arr) 
  
print (checkSubarraySum(arr, n, k)) 