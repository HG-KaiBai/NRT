check several things to me.

1. In testing, the run_cpu will read tau from cases in tuning, not folder hyper_para. But nncde, nrt and deepsurv will read it and other parameters from the folder hyper_para.

2. Check the summary of the testing scripts. I want it save individual json files and then summary data settings table automatically at last.

3. In tuning, let current tuning method for nrt use grid tuning search like nncde instead of random search like deepsurv.

