from os import system, remove, path

if __name__ == "__main__":
    OUT_FILENAME = "5_compare_trainings_out.txt"
    
    if path.exists(OUT_FILENAME):
        remove(OUT_FILENAME)
    
    for numParallelThreads in [2, 8, 32, 128, 256]:
        for prefetchBufferSize in [2, 4, 6, 8, 10]:
            system(f"python3 5_train_resnet34.py {numParallelThreads} {prefetchBufferSize}")
            
        with open(OUT_FILENAME, 'a') as outfile:
            outfile.write('~~~~~~~~~~~~~~~\n')