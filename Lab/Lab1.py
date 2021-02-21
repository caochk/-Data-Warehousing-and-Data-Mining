# def mySqrt(x):
#     l, r, ans = 0, x, -1
#     while l <= r:
#         mid = (l + r) // 2
#         if mid * mid <= x:
#             ans = mid
#             l = mid + 1
#         else:
#             r = mid - 1
#     print(ans)
#
# mySqrt(-1)

import math

def mySqrt(x):
    if x < 0:
        return -1

    else:
        left = 0
        right = x
        while left <= right:
            mid = math.floor((left + right)/2)
            if mid ** 2 == x:
                return mid
            elif mid ** 2 < x:
                left = mid + 1
                ans = mid
            else:
                right = mid - 1
        return ans

aaa = mySqrt(13)
print(aaa)