# put highprio here
#echo 'Highprio tasks' | tee logs/test_phase.log
# calls test_all_known w/ tests needed/defined in Papers:JDDC:plots

# python tools/test_all_known.py --tests test_progressive_rawnind --model_types denoise
echo "" > logs/test_phase.log

echo '2. Compress noisy (manproc_bostitch)' | tee -a logs/test_phase.log
python tools/test_all_known.py --tests test_manproc_bostitch --model_types dc --banned_models JPEGXL_linRGB gamma preup DenoiseThenCompress nocc
python tools/test_all_known.py --tests test_manproc_bostitch --model_types dc --banned_models JPEGXL_linRGB gamma preup

echo '2. Compress noisy (manproc)' | tee -a logs/test_phase.log
python tools/test_all_known.py --tests test_manproc test_manproc_gt test_manproc_q99 --model_types dc

# # 2. Compress noisy (manproc)
# echo '2. Compress noisy (manproc)' | tee -a logs/test_phase.log
# python tools/test_all_known.py --tests test_manproc --model_types dc
# compress q995
echo 'compress q995' | tee -a logs/test_phase.log
python tools/test_all_known.py --tests test_manproc_q995 --model_types dc
# # compress clean
# echo 'compress clean' | tee -a logs/test_phase.log
# python tools/test_all_known.py --tests test_manproc_gt --model_types dc
# # compress clean
# echo 'compress q99' | tee -a logs/test_phase.log
# python tools/test_all_known.py --tests test_manproc_q99 --model_types dc
# 1. Denoising
echo '1. Denoising' | tee -a logs/test_phase.log
python tools/test_all_known.py --tests test_manproc --model_types denoise
# 4b. Denoise extrapairs
python tools/test_all_known.py --tests test_manproc_bostitch --model_types denoise
# compress nearly clean
echo 'compress nearly clean' | tee -a logs/test_phase.log
python tools/test_all_known.py --tests test_manproc_hq --model_types dc
# 3b. Compress clean (playraw-manproc)
echo '3b. Compression clean (playraw-manproc)' | tee -a logs/test_phase.log
python tools/test_all_known.py --tests test_manproc_playraw --model_types dc
# 1b. Denoising (progressive)
#echo '1b. Denoising (progressive)' | tee -a logs/test_phase.log
#python tools/test_all_known.py --tests test_progressive_manproc --model_types denoise
#python tools/test_all_known.py --tests test_progressive_manproc_bostitch --model_types denoise
# echo '4b. Denoise extrapairs' | tee -a logs/test_phase.log
# python tools/test_all_known.py --tests test_ext_raw_denoise --model_types denoise
# echo '4b. Denoise extrapairs nikon' | tee -a logs/test_phase.log
# python tools/test_all_known.py --tests test_ext_nik_raw_denoise --model_types denoise




# # 4. Compress extrapairs
# echo '4. Compress extrapairs' | tee -a logs/test_phase.log
# python tools/test_all_known.py --tests test_ext_raw_denoise --model_types dc
# echo '4. Compress extrapairs nikon' | tee -a logs/test_phase.log
# python tools/test_all_known.py --tests test_ext_nik_raw_denoise --model_types dc
# TODO compress extrapairs2


# 3. Compress clean (playraw)
echo '3. Compress clean (playraw)' | tee -a logs/test_phase.log
python tools/test_all_known.py --tests test_playraw --model_types dc

# finishing dc bostitch
echo 'finishing dc bostitch' | tee -a logs/test_phase.log
python tools/test_all_known.py --tests test_manproc_bostitch --model_types dc 
# everything
#echo 'everything' | tee -a logs/test_phase.log
#python tools/test_all_known.py
echo 'done' | tee -a logs/test_phase.log
