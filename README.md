# ML
用于记录CS229学习的过程

-   以周为单位,在每周的`README`中放置当周的学习计划;
-   每周相关的学习资源,会放到对应的文件夹内
-   如果相应的学习资源话题相关性不是很强,可以放在根目录单独建一个文件夹
-   每周的学习笔记和作业也会放到对应的文件夹下,为了方便,这里没有使用多个`branch`,而是在对应目录下设置子目录,用来标识每个人的作业
-   在学习每个`lecture`时候遇到的问题,通过`issue`提出,尽量同一个主题的使用同一个`issue`
-   关于提交,最好每天提交一次,可以在自己的笔记目录下单独建一个`record`文件记录当日学习总结,或者通过`commit`的`message`来记录
-   目前尚处于起步阶段,很多约定略显稚嫩,后面会不断学习完善的

Issue和discussion提问题的命名规范，方便查看和查找
【Note/Lecture】2.2 问题简述



## 利用分支同步`repo`

尝试通过`branch`来共享我们做的笔记和编程习题.

1.   先将这个repo`clone`到本地;
2.   在本地通过`git branch <branch-name>`创建一个分支;
3.   然后通过`git checkout <branch-name>`切换到你所在的分支
4.   根据上面的约定在每周的文件夹下更新学习任务计划,创建`笔记`和`HW`下的属于你的文件夹,在对应的文件夹下添加笔记和编程作业
5.   添加提交你的修改
6.   切换`git checkout master `切换到主分支,然后  `git merge <branch-name>`j将你分支的变动合并到主分支`master`上. 

## 学习资料

- 杭电MIL实验室2019暑期 研究生机器学习与深度学习入门课程 https://hdumil.github.io/summer-school/
- 《动手学深度学习》李沐线上版pdf https://zh.d2l.ai/

## 课程
- CS229a
   - 课程：https://www.coursera.org/course/ml
   - 资料：https://github.com/fengdu78/Coursera-ML-AndrewNg-Notes

- CS229
   - 课程(Youtube)：https://www.youtube.com/watch?v=jGwO_UgTS7I&list=PLoROMvodv4rMiGQp3WXShtMGgzqpfVfbU
   - 课程(Bilibili)： https://www.bilibili.com/video/BV1JE411w7Ub?spm_id_from=333.337.search-card.all.click
   - 课表：https://cs229.stanford.edu/syllabus-autumn2018.html
   - 资料：https://github.com/maxim5/cs229-2018-autumn
