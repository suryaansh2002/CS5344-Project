if args.model_type == "efficientnet":
    from models.efficientnet_lstm import MultiModalModel
    model = MultiModalModel(tokenizer)
elif args.model_type == "mobilenet":
    from models.mobilenet_roberta import MultiModalModel
    model = MultiModalModel(alpha=0.6)
