## import modules here

################# Question 1 #################
def get_word_dict(taining_data, flag):
    word_dict = dict()
    sms_num = 0
    if flag == 'ham' or flag == 'spam':
        for training_data_ele in taining_data:
            if training_data_ele[1] != flag:
                continue
            else:
                sms_num += 1
                for word, freq in training_data_ele[0].items():
                    if word not in word_dict.keys():
                        word_dict[word] = freq
                    else:
                        word_dict[word] += freq
    else:
        for training_data_ele in taining_data:
            sms_num += 1
            for word, freq in training_data_ele[0].items():
                if word not in word_dict.keys():
                    word_dict[word] = freq
                else:
                    word_dict[word] += freq
    return word_dict, sms_num

def probability_calculation(word_dict, total_word_dict, sms):
    smooth = 1
    prob = 1
    for word in sms:
        if word in total_word_dict.keys():
            if word in word_dict.keys():
                freq_in_wordDict = word_dict[word]
            else:
                freq_in_wordDict = 0

            total_freq_in_wordDict = sum(word_dict.values())

            total_word_num = len(total_word_dict.keys())

            prob *= (freq_in_wordDict + smooth) / (total_freq_in_wordDict + total_word_num)

    return prob


def multinomial_nb(training_data, sms):# do not change the heading of the function
    ham_word_dict, ham_sms_num = get_word_dict(training_data, 'ham')
    spam_word_dict, spam_sms_num = get_word_dict(training_data, 'spam')
    total_word_dict, total_sms_num = get_word_dict(training_data, 'total')

    prob_of_word_in_ham = probability_calculation(ham_word_dict, total_word_dict, sms)
    prob_of_word_in_spam = probability_calculation(spam_word_dict, total_word_dict, sms)

    ratio_of_posterior_prob = ((spam_sms_num * prob_of_word_in_spam) / total_sms_num) / ((ham_sms_num * prob_of_word_in_ham) / total_sms_num)

    return ratio_of_posterior_prob