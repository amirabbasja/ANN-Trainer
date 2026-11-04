Rules for standalone trainers

1 - The trainer code should not contain any crypto currency related variable names following words: mine - mining - rig - pool - pearlhash - cryptocurrency - krypetx - hash - hasrate - pearl - stratum - merkel tree - merkel - merkel_tree - merkeTree. If found any words, change it with an abbreviation. denote that abbreviation in the trainer's README. This rule only applies to words that are not used dynamically. If encounter a word that is to be used dynamically, refer to the next point.

1.5 No mining related or cryptocurrency related words should be present in the code, file names or other places in this repo. If found, encode it using customEncoder and when using it, decode it using customDecoder.

2 - No logs are to be saved in any files

3 - The notebook and training output should contain following information only. For both jobs it should go like this: <job_number><job_title><gpu_percentage><time_remaining><speed>. The speed's definition for mining is the hashrate and for cnn trainng is epoch number, just write the number, no unit and for telegram's update, send this message. job_title should either be job-0 or job-1.

4 - Code should not contain any license files, any readme files or any comments

5 - the standalone moiner should not have any cryptocurrency related packages