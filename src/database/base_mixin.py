from abc import ABC, abstractmethod

class BaseMixin(ABC):
    def __init__(self):
        super().__init__()
        self.conn = self.get_connection()
        self.cur = self.conn.cursor()
        self.logger = self.get_logger()
        self.sqla_engine = self.get_sqla_engine()

    @abstractmethod
    def get_connection(self):
        """Subclass must provide database client"""
        pass

    @abstractmethod
    def get_logger(self):
        """Subclass must provide logger"""
        pass

    @abstractmethod
    def get_sqla_engine(self):
        """Subclass must provide logger"""
        pass