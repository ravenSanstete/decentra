# generate typical topology file
import logging
logging.basicConfig(format='%(asctime)s %(message)s', level=logging.DEBUG)



def generate_chain(num):
    path = "topo_{}_chain.txt".format(num)
    logging.info("Generate Chain of length {} in {}".format(num, path))
    f = open(path, 'w+')
    f.write(str(num)+"\n")
    for i in range(num-1):
        f.write("{} {}\n".format(i+1, i))
    f.close()




if __name__ == "__main__":
    generate_chain(50)
    
    
