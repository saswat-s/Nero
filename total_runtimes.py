import numpy as np

nero = [.03, .03, .03, .04,  # Family
        .02, .03, .03, .03,  # Lympoho
        .03, .03, .03, .03,  # Biopax
        .12, .12, .13, .14,  # Mutagenesis
        .25, .20, .21, .21  # Carcinogenesis
        ]

celoe = [5.70, 6.15, 6.27, 6.10,  # Family
         4.87, 6.70, 6.75, 6.37,  # Lympoho
         7.54, 6.94, 7.20, 6.96,  # Biopax
         13.81, 13.97, 14.65, 14.36,  # Mutagenesis
         25.19, 25.30, 25.44, 25.73  # Carcinogenesis
         ]

eltl = [3.36, 3.31, 3.38, 3.07,  # Family
        3.41, 3.63, 3.77, 3.39,  # Lympoho
        4.87, 4.10, 4.34, 4.02,  # Biopax
        10.64, 10.73, 11.74, 10.48,  # Mutagenesis
        20.73, 20.58, 20.84, 21.45  # Carcinogenesis
        ]

lp = 50
nero = np.array(nero) * lp
celoe = np.array(celoe) * lp
eltl = np.array(eltl) * lp

print(np.sum(nero) / 60)
print(np.sum(celoe) / 60)
print(np.sum(eltl) / 60)
