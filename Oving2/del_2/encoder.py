import emoji as e


#  Makes an alphabet based on the words in the list
def get_alphabet(word_list):
    alphabet = ''

    for word in word_list:
        for letter in word:
            new_letter = True
            for a in alphabet:
                if letter == a:
                    new_letter = False

            if new_letter:
                alphabet += letter

    return alphabet


#  Makes a char encoder for all the letters in the alphabet
def get_char_encoder(alphabet):
    char_encodings = [0] * len(alphabet)
    dict = []

    for letter in range(len(alphabet)):
        dict.append(alphabet[letter])
        row = [0] * len(alphabet)
        row[letter] = 1
        char_encodings[letter] = row

    return char_encodings, dict


#  Takes input ['cat', 'rat'], char_encodings for those words and dict, and returns one-hot encoded x-train
def get_x_train(words_list, char_encodings, dict):
    x_train = [0] * len(words_list)

    for i in range(len(words_list)):

        letter_row = []

        for letter in words_list[i]:

            for j in range(len(dict)):
                if letter == dict[j]:
                    letter_row.append(char_encodings[j])

        x_train[i] = letter_row

    return x_train


#  Takes [':rat:', ':cat:'] and returns encoding table for emojis and dict
def get_emoji_encodings(emojis_list):
    emoji_encodings = [0] * len(emojis_list)
    emoji_dict = []

    for i in range(len(emojis_list)):
        emoji_dict.append(e.emojize(emojis_list[i]))
        emoji_row = [0] * len(emojis_list)
        emoji_row[i] = 1
        emoji_encodings[i] = emoji_row

    return emoji_encodings, emoji_dict
