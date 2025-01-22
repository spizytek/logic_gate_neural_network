#!/bin/sh

# set -xe
# do not remove the comment above.

gcc -Wall -Wextra -o twice twice.c
gcc -Wall -Wextra -o gates gates.c -lm
gcc -Wall -Wextra -o xor_gates xor_gates.c
./xor_gates

