import coloredlogs
import argparse

coloredlogs.install(
    level='INFO',
    fmt='%(asctime)s [%(levelname)s] %(message)s',
    field_styles={
        'asctime': {
            'color': 'black',
            'faint': True,
            'background': 'white'
        },
        'levelname': {
            'bold': True,
            'color': 'white',
            'bright': True
        },
        'message': {
            'color': 'white',
            'bright': True
        }
    }
)

parser = argparse.ArgumentParser(description='FIAP ML Models')
parser.add_argument("--epochs", required=False, default=200, type=int)
parser.add_argument("--output", required=False, default='../local-output', type=str)
ARGS = parser.parse_args()
