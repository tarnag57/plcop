import tensorflow as tf


'''
This is a class containing all the information about the training process
(e.g. models, optimizer, metadata, ...)
The class follows the singleton design pattern
'''


class ModelContext():

    __instance = None

    @staticmethod
    def create_context(
        args,
        optimizer,
        checkpoint,
        tokenizer,
        seq_to_seq_model
    ):
        ModelContext(
            args,
            optimizer,
            checkpoint,
            tokenizer,
            seq_to_seq_model
        )

    @staticmethod
    def add_datset(
        all_clauses_tensor,
        train_dataset,
        train_input,
        val_dataset,
        val_input
    ):
        if ModelContext.__instance == None:
            raise Exception("Model context not yet initialised")

        if ModelContext.__instance.train_dataset != None:
            raise Exception("Datasets already added to the context")

        else:
            ctx = ModelContext.__instance
            ctx.all_clauses_tensor = all_clauses_tensor
            ctx.train_dataset = train_dataset
            ctx.train_input = train_input
            ctx.train_size = len(train_input)
            ctx.val_dataset = val_dataset
            ctx.val_input = val_input
            ctx.val_size = len(val_input)
            ctx.steps_per_epoch = len(train_input) // ctx.args.batch_size
            ctx.val_steps_per_epoch = len(val_input) // ctx.args.batch_size

    @ staticmethod
    def get_context():
        if ModelContext.__instance != None:
            return ModelContext.__instance
        else:
            raise Exception("Please initialise the context before using")

    def __init__(
        self,
        args,
        optimizer,
        checkpoint,
        tokenizer,
        seq_to_seq_model
    ):
        if ModelContext.__instance != None:
            raise Exception(
                "The context is a singleton; the constructor should never be called")
        else:
            self.args = args
            self.optimizer = optimizer
            self.checkpoint = checkpoint
            self.tokenizer = tokenizer
            self.seq_to_seq_model = seq_to_seq_model

            # These might be initialised later
            self.all_clauses_tensor = None
            self.train_dataset = None
            self.train_input = None
            self.train_size = None
            self.val_dataset = None
            self.val_input = None
            self.val_size = None
            self.steps_per_epoch = None
            self.val_steps_per_epoch = None

            ModelContext.__instance = self
