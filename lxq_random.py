import random

# 定义字符集
chars = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789!@#$%^&*()_+-=[]{};:,.<>?`~"

# 生成随机8位字符串
random_string = ''.join(random.choices(chars, k=8))

print(random_string)