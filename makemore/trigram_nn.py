import torch
import matplotlib.pyplot as plt

words = open('names.txt', 'r').read().splitlines()[:8]
print(words)

letters = sorted(list(set(''.join(words))))
words = ('.' + word + '.' for word in words)
trigrams = [(word[i:i+2], word[i+2]) for word in words for i in range(len(word)) if i+2 < len(word)]
letter_pairs = set(tup[0] for tup in trigrams)
print(trigrams)

str_to_inx_pairs = {str:inx+1 for inx, str in enumerate(letter_pairs)}
str_to_inx_letter = {str:inx+1 for inx, str in enumerate(letters)}
str_to_inx_pairs['.'] = 0
str_to_inx_letter['.'] = 0
inx_to_str_pairs = {inx:str for str, inx in str_to_inx_pairs.items()}
inx_to_str_letter = {inx:str for str, inx in str_to_inx_letter.items()}
print(str_to_inx_letter)
print(str_to_inx_pairs)

N = torch.zeros(len(letters)+1, len(letter_pairs)+1, dtype=torch.int32)
for trigram in trigrams:
    i = str_to_inx_letter[trigram[1]]
    j = str_to_inx_pairs[trigram[0]]
    N[i, j] += 1


print(letters)
plt.figure(figsize=(15,13))
plt.imshow(N, cmap='Blues')
for i in range(len(letters) + 1):
    for j in range(len(letter_pairs) + 1):
        # print(inx_to_str_letter[i])
        chstr = inx_to_str_pairs[j] + inx_to_str_letter[i]
        # print(chstr)
        plt.text(j, i, chstr, ha="center", va="bottom", color='gray')
        plt.text(j, i, N[i, j].item(), ha="center", va="top", color='gray')
plt.axis('off')


# print(N)
# plt.figure(figsize=(16,16))
# plt.imshow(N)

plt.show()
