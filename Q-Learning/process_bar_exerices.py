"""

This is a test program aim to learn to use function "print"
The ustage is make a process bar

"""
import time

bar_content='-'
process_bar=50*bar_content
# print(process_bar)

for i in range(100):
    if i%2==0:
        process_bar=process_bar.replace(process_bar[int(i/2)],'>',1)
    print("\r",process_bar,i,'%',end="",flush=True)
    time.sleep(0.3)
    i+=1

