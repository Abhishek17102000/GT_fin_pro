from lib.dataset import Dataset2D
from lib.core.config import PENNACTION_DIR


class PennAction(Dataset2D):
    def __init__(self, seqlen, overlap=0.75, debug=False):
        db_name = 'pennaction'

        super(PennAction, self).__init__(
            seqlen = seqlen,
            folder=PENNACTION_DIR,
            dataset_name=db_name,
            debug=debug,
            overlap=overlap,
        )
        print(f'{db_name} - number of dataset objects {self.__len__()}')
