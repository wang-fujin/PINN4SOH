import pyautogui as pg
import keyboard

def rgb2hex(r, g, b):
    return '#{:02x}{:02x}{:02x}'.format(r, g, b)

def hex_to_decimal(hex_color):
    # 去除可能包含在颜色值前的'#'符号
    # Remove '#' symbols that may be included in front of color values
    if hex_color.startswith('#'):
        hex_color = hex_color[1:]

    # 将颜色值分成红色、绿色和蓝色部分
    # Divide the color value into red, green, and blue
    red_hex = hex_color[0:2]
    green_hex = hex_color[2:4]
    blue_hex = hex_color[4:6]

    # 将16进制的颜色值转换为10进制数值
    # Convert hexadecimal color values to decimal values
    red_decimal = int(red_hex, 16)
    green_decimal = int(green_hex, 16)
    blue_decimal = int(blue_hex, 16)

    # 将10进制数值除以255，得到0-1之间的小数表示
    # Divide the decimal value by 255 to get a decimal representation between 0 and 1
    red_decimal_normalized = red_decimal / 255
    green_decimal_normalized = green_decimal / 255
    blue_decimal_normalized = blue_decimal / 255

    return (red_decimal_normalized, green_decimal_normalized, blue_decimal_normalized)

# 测试(test)
hex_color = "#F7C97E"
result = hex_to_decimal(hex_color)
print(result)

