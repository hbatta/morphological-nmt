# morphological-nmt
Read Me:
#training embeddings(BPE):

spm_train --input=/home/ubuntu/nmt-data/parallel/final_preprocessedfile.txt --model_prefix=spe_word_final2 --vocab_size=1325947 --character_coverage=1.0 --model_type=bpe --split_by_whitespace=true

#Generating SPM (Byte Pair levels encoding):

FAIRSEQ=/home/ubuntu/fairseq/fairseq_cli
MODEL=/home/ubuntu/spe_bpe_final2.model
SPM=/usr/local/bin/spm_encode
SPM_TRAIN=/home/ubuntu/spm_files/train/trn
SPM_VAL=/home/ubuntu/spm_files/val/val
SPM_TST=/home/ubuntu/spm_files/test/tst
TRAINDIR=/home/ubuntu/nmt-data/parallel
VALDIR=/home/ubuntu/nmt-data/dev-test
SRCVAL=dev_o.txt
TARVAL=dev.en
SRCTST=test_o.txt
TARTST=test.en
SRCTR=hi_morp.txt
TARTR=en_morp_final1.txt
${SPM} --model=${MODEL} < ${TRAINDIR}/${SRCTR} > ${SPM_TRAIN}.spm.${SRC} &
${SPM} --model=${MODEL} < ${TRAINDIR}/${TARTR} > ${SPM_TRAIN}.spm.${TGT} &
${SPM} --model=${MODEL} < ${VALDIR}/${SRCVAL} > ${SPM_VAL}.spm.${SRC} &
${SPM} --model=${MODEL} < ${VALDIR}/${TARVAL} > ${SPM_VAL}.spm.${TGT} &
${SPM} --model=${MODEL} < ${VALDIR}/${SRCTST} > ${SPM_TST}.spm.${SRC} &
${SPM} --model=${MODEL} < ${VALDIR}/${TARTST} > ${SPM_TST}.spm.${TGT} &

cut -f1 spe_bpe_final2.vocab | tail -n +4 | sed "s/$/ 100/g" > fairseq.vocab

#Fairseq Preprocessing (Binarizing the data):

NAME=en-hi
DEST=/home/ubuntu/postprocessed
DICT=/home/ubuntu/fairseq.vocab
python ${FAIRSEQ}/preprocess.py \
--source-lang ${SRC} \
--target-lang ${TGT} \
--trainpref ${SPM_TRAIN}.spm \
--validpref ${SPM_VAL}.spm \
--testpref ${SPM_TST}.spm  \
--destdir ${DEST}/${NAME} \
--thresholdtgt 0 \
--thresholdsrc 0 \
--srcdict ${DICT} \
--tgtdict ${DICT} \
--workers 70

#Training


CUDA_VISIBLE_DEVICES=0 fairseq-train \
    /home/ubuntu/postprocessed/en-hi \
    --arch transformer_iitb_hi_en --share-decoder-input-output-embed \
    --optimizer adam --adam-betas '(0.9, 0.98)' --clip-norm 0.0 \
    --lr 5e-4 --lr-scheduler inverse_sqrt --warmup-updates 4000 \
    --dropout 0.3 --weight-decay 0.0001 \
    --criterion label_smoothed_cross_entropy --label-smoothing 0.1 \
    --max-tokens 4096 \
    --eval-bleu \
    --eval-bleu-args '{"beam": 5, "max_len_a": 1.2, "max_len_b": 10}' \
    --eval-bleu-detok moses \
    --eval-bleu-remove-bpe \
    --eval-bleu-print-samples \
    --best-checkpoint-metric bleu --save-dir checkpoints/fconv --maximize-best-checkpoint-metric
