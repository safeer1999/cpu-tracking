from brightness import brightness
import time
  
def checkSubarraySum(arr, n, k):
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
    # print(start)
    # print(end)
    return [start,end]
    # return final_sum

# Driver code


def activate(arr):

    k = len(arr)//3
    n = len(arr)
    ranges=(checkSubarraySum(arr, n, k))
    for i in range(len(arr)):
        print(arr[i])
        time.sleep(1)
        if i==ranges[0]:
            brightness(100,verbose=True)
        elif i==ranges[1]:
            brightness(10,verbose=True)

def main():
    activate()

if __name__ == "__main__":
    main()
