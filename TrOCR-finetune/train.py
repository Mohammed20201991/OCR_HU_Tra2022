import fun
import unit_test
fun.os.environ['CUDA_VISIBLE_DEVICES'] = '5'
fun.os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

from transformers import DistilBertTokenizer

# from transformers import AutoTokenizer
fun.processor.tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-cased')
#AutoTokenizer.from_pretrained("distilbert-base-multilingual-cased")
def main():
    df = fun.load_laia()
    # df = load_dataset()
    print(df.head())
    train_dataset, eval_dataset = fun.create_datasets(df)

    print("Number of training examples:", len(train_dataset))
    print("Number of validation examples:", len(eval_dataset))

    
    # processor.tokenizer.pad_token = processor.tokenizer.eos_token
    encoding = train_dataset[0]
    for k, v in encoding.items():
        print(k, v.shape)

    labels = encoding['labels']
    labels[labels == -100] = fun.processor.tokenizer.pad_token_id
    label_str = fun.processor.decode(labels, skip_special_tokens=True)
    print(label_str)

    # print('step1')
    model = fun.VisionEncoderDecoderModel.from_encoder_decoder_pretrained("google/vit-base-patch16-384", "bert-base-multilingual-uncased")

    # set special tokens used for creating the decoder_input_ids from the labels
    model.config.decoder_start_token_id = fun.processor.tokenizer.cls_token_id
    model.config.pad_token_id = fun.processor.tokenizer.pad_token_id
    # make sure vocab size is set correctly
    model.config.vocab_size = model.config.decoder.vocab_size

    # set beam search parameters
    model.config.eos_token_id = fun.processor.tokenizer.sep_token_id
    model.config.max_length = 64
    model.config.early_stopping = True
    model.config.no_repeat_ngram_size = 3
    model.config.length_penalty = 2.0
    model.config.num_beams = 4

    
   
    training_args = fun.Seq2SeqTrainingArguments(
        dataloader_num_workers=0,
        num_train_epochs=12,
        learning_rate=2e-5,
        predict_with_generate=True,
        evaluation_strategy="steps",
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        fp16=False,
        output_dir=f'models/ViT_distilbert/{fun.datetime.now().strftime("%Y%m%d%H%M%S")}',
        logging_steps=100,
        save_steps=2000,
        eval_steps=200,
    )

    # instantiate trainer
    trainer = fun.Seq2SeqTrainer(
        model=model,
        tokenizer=fun.processor.feature_extractor,
        args=training_args,
        compute_metrics=fun.compute_metrics,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=fun.default_data_collator,
    )
    trainer.train()


if __name__ == '__main__':
    main()