# Group similar cattle IDs into consolidated classes (example mapping)
CLASS_MAPPING = {
    **{i: 0 for i in range(0, 20)},      # Group 0-19 as "Young Cows"
    **{i: 1 for i in range(20, 50)},     # Group 20-49 as "Mature Cows"
    **{i: 2 for i in range(50, 100)},    # Group 50-99 as "Bull Group A"
    **{i: 3 for i in range(100, 120)},
    **{i: 4 for i in range(120, 140)},
    **{i: 5 for i in range(140, 160)},
    **{i: 6 for i in range(160, 180)},
    **{i: 7 for i in range(180, 200)},
    **{i: 8 for i in range(200, 220)},
    **{i: 9 for i in range(220, 240)},
    **{i: 10 for i in range(240, 260)},
    **{i: 11 for i in range(260, 280)},
    **{i: 12 for i in range(280, 300)},
    **{i: 13 for i in range(300, 320)},
    **{i: 14 for i in range(320, 340)},
    **{i: 15 for i in range(340, 350)},
    **{i: 16 for i in range(350, 360)},
    **{i: 17 for i in range(360, 370)},
    **{i: 18 for i in range(370, 380)},
    **{i: 19 for i in range(380, 384)},  # Final group
}


def get_num_classes():
    """
    Get the number of classes from the mapping programmatically.
    Returns the count of unique mapped class values.
    """
    return len(set(CLASS_MAPPING.values()))
