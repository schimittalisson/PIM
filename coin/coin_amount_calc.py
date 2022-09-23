"""
Coin recognition, real life application
task: calculate the value of coins on picture
"""

import cv2
import numpy as np

def detect_coins():
    coins = cv2.imread('/home/alisson/Documentos/udesc/PIM/coin/coins.jpg', 1)

    gray = cv2.cvtColor(coins, cv2.COLOR_BGR2GRAY)
    img = cv2.medianBlur(gray, 7)
    circles = cv2.HoughCircles(
        img,  # source image
        cv2.HOUGH_GRADIENT,  # type of detection
        1,
        50,
        param1=100,
        param2=50,
        minRadius=10,  # minimal radius
        maxRadius=380,  # max radius
    )

    coins_copy = coins.copy()


    for detected_circle in circles[0]:
        x_coor, y_coor, detected_radius = detected_circle
        coins_detected = cv2.circle(
            coins_copy,
            (int(x_coor), int(y_coor)),
            int(detected_radius),
            (0, 255, 0),
            4,
        )

    cv2.imwrite("/home/alisson/Documentos/udesc/PIM/coin/coin_test.jpg", coins_detected)

    return circles

def calculate_amount():
    reais = {
        "0.1 BRL": {
            "value": 0.1,
            "radius": 20,
            "ratio": 1,
            "count": 0,
        },
        "0.5 BRL": {
            "value": 0.5,
            "radius": 23,
            "ratio": 1.15,
            "count": 0,
        },
        "1 BRL": {
            "value": 1,
            "radius": 27,
            "ratio": 1.34,
            "count": 0,
        },
    }

    circles = detect_coins()
    radius = []
    coordinates = []

    for detected_circle in circles[0]:
        x_coor, y_coor, detected_radius = detected_circle
        radius.append(detected_radius)
        coordinates.append([x_coor, y_coor])

    smallest = min(radius)
    tolerance = 0.04
    total_amount = 0

    coins_circled = cv2.imread('/home/alisson/Documentos/udesc/PIM/coin/coin_test.jpg', 1)
    font = cv2.FONT_HERSHEY_SIMPLEX

    for coin in circles[0]:
        ratio_to_check = coin[2] / smallest
        coor_x = coin[0]
        coor_y = coin[1]
        for real in reais:
            value = reais[real]['value']
            if abs(ratio_to_check - reais[real]['ratio']) <= tolerance:
                reais[real]['count'] += 1
                total_amount += reais[real]['value']
                cv2.putText(coins_circled, str(value), (int(coor_x), int(coor_y)), font, 1,
                            (0, 0, 0), 4)

    print(f"O valor total é de: {total_amount} BRL")
    for real in reais:
        pieces = reais[real]['count']
        print(f"{real} = {pieces}x")


    cv2.imwrite('/home/alisson/Documentos/udesc/PIM/coin/coins_value.jpg', coins_circled)



if __name__ == "__main__":
    calculate_amount()