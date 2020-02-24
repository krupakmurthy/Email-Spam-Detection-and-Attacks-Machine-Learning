# Email-Spam-Detection-and-Attacks-Machine-Learning

> For detailed methodlogy,steps followed and results,  please refer to "MidTermProjectReport.pdf" and "FinalProjectReport.pdf"

Preprocessed data without lemmatization and removal of stop words (“data clean”) : data_clean.py
Preprocessed data with lemmatization (“data clean lemmatize”): data_clean_lemmatize.py
Preprocessed data with all the processes(refer MidTermprojectReport.pdf) (“data clean stop words”): data_clean_stop_words.py

For creating a dictionary, run create dictionary.py

Test dictionary attacks:run train_test_data_clean_Dictionary.py
Test frequency attacks:run train_test_data_clean_Frequency.py
Test frequency ratio attacks:run train_test_data_clean_Frequency_Ratio.py

Find barely legitimate email: find_witness_first.py
Finds the number of good words from a dictionary which when added can turn the bare-legitimate email to non-spam email: first.py
