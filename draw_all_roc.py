import os.path
import sys
import numpy as np
from six.moves import xrange
import matplotlib.pyplot as plt
colors = ['#008000','#000000','#0000FF','#FF8C00','#8B0000','#4B0082','#8FBC8F','#483D8B','#2F4F4F','#B8860B','#556B2F','#9932CC']
markers = ['.','o','v' ,'*','H','X' ,'d','s','p','1','>','<']
'''
burlywood':            '#DEB887',
'cadetblue':            '#5F9EA0',
'chartreuse':           '#7FFF00',
'chocolate':            '#D2691E',
'coral':                '#FF7F50',
'cornflowerblue':       '#6495ED',
'cornsilk':             '#FFF8DC',
'crimson':              '#DC143C',
'cyan':                 '#00FFFF',
'darkblue':             ,
'darkcyan':             '#008B8B',
'darkgoldenrod':        '#B8860B',
'darkgray':             '#A9A9A9',
'darkgreen':            ,
'darkkhaki':            '#BDB76B',
'darkmagenta':          '#8B008B',
'darkolivegreen':       '#556B2F',
'darkorange':           '#FF8C00',
'darkorchid':           '#9932CC',
'darkred':              '#8B0000',
'darkseagreen':         '#8FBC8F',
'pink':                 '#FFC0CB',
'gray':                 '#808080',
'green':                '#008000',
'bisque':               '#FFE4C4',
'black':                '#000000',
'blanchedalmond':       '#FFEBCD',
'blue':                 '#0000FF',
'brown':                '#A52A2A',

'darkorange':           '#FF8C00',

'darkred':              '#8B0000',

'deeppink':             '#FF1493',
'darksalmon':           '#E9967A',
'darkseagreen':         '#8FBC8F',
'darkslateblue':        '#483D8B',
'darkslategray':        '#2F4F4F',
}
'.'       point marker
','       pixel marker
'o'       circle marker
'v'       triangle_down marker
'^'       triangle_up marker
'<'       triangle_left marker
'>'       triangle_right marker
'1'       tri_down marker
'2'       tri_up marker
'3'       tri_left marker
'4'       tri_right marker
's'       square marker
'p'       pentagon marker
'*'       star marker
'h'       hexagon1 marker
'H'       hexagon2 marker
'+'       plus marker
'x'       x marker
'D'       diamond marker
'd'       thin_diamond marker
'|'       vline marker
'_'       hline marker
'''
data_dir = '/home/zhangs/tensorflow-master/tensorflow/examples/speech_lyc/tmp/fa_fr_data'
print("path exist? = " + str(os.path.exists(data_dir)))
dirname = os.listdir(data_dir)
model_dir=[]
pic = []
print(dirname)
for i,name in enumerate(dirname):
    filepath = os.path.join(data_dir,name)
    if os.path.isdir(filepath):
        model_dir.append(dirname[i])
print ("find model = " + str(model_dir))
print("find model numbers= " + str(len(model_dir)))
plt.xlabel('False Alarm rate')
plt.ylabel('False reject rate')
plt.title(' ROC picture')
#plt.ylim(0, 0.4)
#plt.xlim(0, 0.01)
color = 0
model_dir.sort()
for model in model_dir:
    csv_path = data_dir + '/'+model +'/'+'fafr_data.csv'
    if os.path.exists(csv_path):
        fa,fr= np.loadtxt(csv_path,delimiter=',')
        fa = fa[::20]
        fr = fr[::20]
        line, = plt.plot(fa, fr, label='FA  Vs. FR ', linewidth=1.5,marker=markers[color],linestyle='--',color=colors[color])  # linestyle='--',marker=markers[color]
        pic.append(line)
        color += 1
        #print(fa.shape)
    else:
        print("NO data path!")
plt.ylim(0, 0.4)
#plt.xlim(0, 0.003)
plt.legend(handles=pic,labels = model_dir,loc='upper right',markerscale=1)
plt.show()
'''
plt.plot(list(x_plot), y_plot, label='FA  Vs. FR ', linewidth=1, color='red')  # san dian

'''
