# s = "abcabcbb"

# wins = []
# max_len = -1
# for i in range(len(s)):
#   if s[i] not in wins:
#     wins.append(s[i])
#     max_len = max(max_len, len(wins))
#   else:
#     for j in range(len(wins)):
#       if wins[j] != s[i]:
#         wins.pop(0)
#         break
      
# print(max_len)

import torch
print(torch.__version__)