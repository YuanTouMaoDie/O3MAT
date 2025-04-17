# O3MAT

Project: O3MAT

图：Map用来绘制六中Mehod中的各自浓度地形图和对应的差值浓度分布

Scatter 是散点密度图

图名和文件夹均为自动读取闯将

数据融合采用并且处理实现规定区域的融合

小时脚本:	  DataFusionHourly_   +CV（验证） +Batch（批量化运行） +Parrel（并行运行）+Tz (时区调整)

日脚本：DataFusionDaily_   +CV（验证） +Batch（批量化运行） +Parrel（并行运行）

画图脚本：Map_  +Diff (差值图)  +Point(只画点)  +PointWithGrid(点网叠加)

处理工具：Tool_  

output: Picture:{Year}_{Metrcis: W126}__{Use: AloneMap/CompareMap/ CompareScatter/}

Data:{Year}_{Metrcis: Data}__{Use: CV/WithoutCV }

OHTHER: Merged/Test/TestMap For Test

Region: Monitor Timeozone Or CONUS
