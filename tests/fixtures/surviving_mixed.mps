NAME          SURVMIX
ROWS
 N  OBJ
 G  R1
 L  R2
 E  R3
COLUMNS
    X1        OBJ        1.0   R1           1.0
    X1        R2         2.0   R3           1.0
    X2        OBJ        2.0   R1           1.0
    X2        R2         1.0   R3          -1.0
    X3        OBJ        0.5   R1           1.0
    X3        R2        -1.0   R3           2.0
    X4        OBJ        3.0   R1          -1.0
    X4        R2         1.0   R3           1.0
RHS
    RHS1      R1         2.0   R2           5.0
    RHS1      R3         3.0
BOUNDS
 UP BND1      X1         4.0
 LO BND1      X2         0.0
 FR BND1      X3
 LO BND1      X4         0.0
ENDATA
