import coloredlogs

coloredlogs.install(
    level='INFO',
    fmt='%(asctime)s [%(levelname)s] %(message)s',
    field_styles={
        'asctime': {'color': 'white', 'normal': True},
        'levelname': {'bold': True, 'color': 'white', 'bright': True},
        'message': {'color': 'white', 'bright': True}
    }
)