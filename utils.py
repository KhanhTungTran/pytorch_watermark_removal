 
class COLORS:
    GREY = 'GREY'
    RED = 'RED'
    GREEN = 'GREEN'
    YELLOW = 'YELLOW'
    BLUE = 'BLUE'
    PINK = 'PINK'
    BEIGE = 'BEIGE'


TEXT_COLOR = {}
TEXT_COLOR[COLORS.GREY] = '\33[90m'
TEXT_COLOR[COLORS.RED] = '\33[91m'
TEXT_COLOR[COLORS.GREEN] = '\33[92m'
TEXT_COLOR[COLORS.YELLOW] = '\33[93m'
TEXT_COLOR[COLORS.BLUE] = '\33[94m'
TEXT_COLOR[COLORS.PINK] = '\33[95m'
TEXT_COLOR[COLORS.BEIGE] = '\33[96m'

BG_COLOR = {}
BG_COLOR[COLORS.GREY] = '\33[100m'
BG_COLOR[COLORS.RED] = '\33[41m'
BG_COLOR[COLORS.GREEN] = '\33[42m'
BG_COLOR[COLORS.YELLOW] = '\33[43m'
BG_COLOR[COLORS.BLUE] = '\33[44m'
BG_COLOR[COLORS.PINK] = '\33[45m'
BG_COLOR[COLORS.BEIGE] = '\33[46m'


def printFixed(value, end='', color='', bgColor='', width=8):
    strValue = f'{value}'.center(width)
    if color != '':
        strValue = TEXT_COLOR[color] + strValue + '\033[0m'
    if bgColor != '':
        strValue = BG_COLOR[bgColor] + strValue + '\033[0m'
    print(strValue, end=end)
