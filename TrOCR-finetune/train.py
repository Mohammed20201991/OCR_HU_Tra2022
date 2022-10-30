import fun
# modifying  the tokenizer 
fun.processor.tokenizer = fun.AutoTokenizer.from_pretrained(fun.Decoder)

def main():
    df = fun.load_laia()
    # df = load_dataset()
    print(df.head(4))
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
    
    model = fun.VisionEncoderDecoderModel.from_encoder_decoder_pretrained(fun.Encoder,fun.Decoder)
    # setting model configuration
    configured_model = fun.trocr_model_config(model)

    training_args = fun.Seq2SeqTrainingArguments(
        predict_with_generate=True,
        evaluation_strategy="steps",
        learning_rate=2e-5,
        num_train_epochs=12,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        fp16=True,
        output_dir="./vit_bert",
        logging_steps=100,
        save_steps=1000,
        eval_steps=500,
    )

    # instantiate trainer
    trainer = fun.Seq2SeqTrainer(
        model=configured_model,
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


