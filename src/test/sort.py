
# 从小到大排列
def llz1(arr):
    n = len(arr)
    for i in range(n - 1):
        for j in range(n - 1 - i):
            if arr[j] > arr[j + 1]:
                arr[j], arr[j + 1] = arr[j + 1], arr[j]
    return arr

# 从大到小排列
def llz2(arr):
    n = len(arr)
    for i in range(n - 1):
        for j in range(n - 1 - i):
            if arr[j] < arr[j + 1]:
                arr[j], arr[j + 1] = arr[j + 1], arr[j]
    return arr

# 1. 接收用户输入
# "8 0 3 7 1 5"
user_input = input("请输入一组数字，用空格分隔：")


# 2. 将输入的字符串转换为整数列表
# "8 0 3 7 1 5" -> "8" "0" "3" "7" "1" "5" -> 8 0 3 7 1 5 -> [8, 0, 3, 7, 1, 5]
data = list(map(int, user_input.split()))

print("排序前:", data)

sorted_data = llz1(data)

print("排序后:", sorted_data)
