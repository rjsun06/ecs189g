
voc=dict.fromkeys([])
voci=dict.fromkeys([])


def MyRead(path):
    count = 1
    set=[]
    with open(path,'rb') as f:
        lines = f.readlines()
        for line in lines[1:]:
            line=str(line)
            print(line)
            i=1
            while(line[i]!=','):
                print(line[i])
                i=i+1
            newline=[]
            print(line[i+1:-4].replace('"','').replace('\'','').replace('?','').replace('!','').replace(',','').replace('.','').replace('[','').replace(']','').replace('\\',''))
            for word in line[i+1:-4].replace('"','').replace('\'','').replace('?','').replace('!','').replace(',','').replace('.','').replace('[','').replace(']','').replace('\\','').split(' '):
                voc.setdefault(word,count)
                if voc[word]==count:
                    voci[count]=word
                    count+=1
                    newline.append(voc[word])
            set.append(newline)

    return set

set=MyRead('C:/cygwin64/home/17931/workspace/ecs189G/ECS189G_Winter_2022_Source_Code_Template/data/stage_4_data/text_generation/data')
print(set)
print(voc)
print(voci)