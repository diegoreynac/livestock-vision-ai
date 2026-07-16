from logger import setup_logger

from dataset import read_dataset

logger = setup_logger()

logger.info("Dataset analysis started.")

df = read_dataset()

logger.info("Dataset loaded successfully.")

logger.info(df.head())

logger.info("Finished.")