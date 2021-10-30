#command to get difference between two text files of same type, Run in terminal only
comm -23 <(sort Pisces_40ID_A_CHAIN_ID_only.txt) <(sort 13_extracted_ID_check_list.txt) > 18_difference_A_chain_IDs.txt
