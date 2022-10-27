import sys
sys.path.append('/home/ngyongyossy/mohammad/asdf/TrOCR-finetune/')
import fun
import unit_test

fun.os.environ['CUDA_LAUNCH_BLOCKING'] = "4"

fun.processor.tokenizer = fun.AutoTokenizer.from_pretrained("SzegedAI/charmen-electra")

def main():
  
    df = fun.load_laia()
    # df = load_dataset()
    print(df.head())
    train_dataset, eval_dataset = fun.create_datasets(df)

    print("Number of training examples:", len(train_dataset))
    print("Number of validation examples:", len(eval_dataset))

    encoding = train_dataset[0]
    for k,v in encoding.items():
        print(k, v.shape)

    labels = encoding['labels']
    labels[labels == -100] = fun.processor.tokenizer.pad_token_id
    label_str = fun.processor.decode(labels, skip_special_tokens=True)
    print(label_str)
    
    model = fun.VisionEncoderDecoderModel.from_encoder_decoder_pretrained("google/vit-base-patch16-384","SzegedAI/charmen-electra", trust_remote_code=True)
    # set decoder config to causal lm
    model.config.decoder.is_decoder = True
    model.config.decoder.add_cross_attention = True

    #model.decoder.__dict__

    # set special tokens used for creating the decoder_input_ids from the labels
    model.config.decoder_start_token_id = fun.processor.tokenizer.cls_token_id
    assert model.config.decoder_start_token_id == fun.processor.tokenizer.cls_token_id
    model.config.pad_token_id = fun.processor.tokenizer.pad_token_id
    # make sure vocab size is set correctly
    model.config.vocab_size = model.config.decoder.vocab_size

    # set beam search parameters
    model.config.eos_token_id = fun.processor.tokenizer.sep_token_id
    model.config.max_length = 128
    model.config.early_stopping = True
    model.config.no_repeat_ngram_size = 3
    model.config.length_penalty = 2.0
    model.config.num_beams = 4

    # from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments

    training_args = fun.Seq2SeqTrainingArguments(
        predict_with_generate=True,
        evaluation_strategy="steps",
        learning_rate=2e-5,
        num_train_epochs=12,
        per_device_train_batch_size=24,
        per_device_eval_batch_size=24,
        fp16=True,
        output_dir="./vit_electraHu",
        logging_steps=100,
        save_steps=1000,
        eval_steps=500,
    )

    # instantiate trainer
    trainer = fun.Seq2SeqTrainer(
        model=model,
        tokenizer= fun.processor.feature_extractor,
        args=training_args,
        compute_metrics= fun.compute_metrics,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator= fun.default_data_collator,
    )

    trainer.train()

if __name__ == '__main__':
    main()
