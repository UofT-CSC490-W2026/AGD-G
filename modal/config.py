from enum import Enum

BUCKET='agd-dev-tyson'
AWS_REGION='ca-central-1'
RDS_HOST='agd-dev-postgres.cdsyi46ammw7.ca-central-1.rds.amazonaws.com'
RDS_PORT=5432
RDS_USER='modal_user'
IMAGE_PREFIX='samples/'
IMAGE_POSTFIX='.png'

class GraphType(Enum):
    THREE_D = 1
    AREA = 2
    BAR = 3
    BOX = 4
    CANDLE = 5
    HEATMAP = 6
    LINE = 7
    NODE = 8
    OTHER = 9
    PIE = 10
    RADAR = 11
    SCATTER = 12
    TREEMAP = 13

    def __str__(self):
        return self.name

    @staticmethod
    def get_names():
        """
        Get names in the enum format PostgreSQL expects, like
        'FOO', 'BAR', 'BAZ'
        """
        return ", ".join(f"'{member.name}'" for member in GraphType)

