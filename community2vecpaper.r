#######################################
#
# Program to analyze distance between
# Reddit subreddits using the cooccurrence
# of commentors across subreddits. 
# Also implements "subreddit algebra"
# by adding and subtracting subreddit
# vectors. 
# By @martintrevor_ for 538
#
#######################################

require(reshape2)
require(lsa)
require(ggplot2)
require(grid)
require(ggrepel)
require(text2vec)
require(Matrix)
require(Rtsne)
require(parallel)
options(mc.cores=5)

##### Part 0: Formatted and processed data in BigQuery

## Creating list of number of users in each subreddit: 
## Thanks to Reddit users /u/Stuck_In_the_Matrix for pulling the data originally and /u/fhoffa for hosting the data on BigQery
#SELECT subreddit, authors, DENSE_RANK() OVER (ORDER BY authors DESC) AS rank_authors
#FROM (SELECT subreddit, SUM(1) as authors
#      FROM (SELECT subreddit, author, COUNT(1) as cnt 
#          FROM [fh-bigquery:reddit_comments.all_starting_201501]
#          WHERE author NOT IN (SELECT author FROM [fh-bigquery:reddit_comments.bots_201505])
#          GROUP BY subreddit, author HAVING cnt > 0)
#      GROUP BY subreddit) t
#ORDER BY authors DESC;

## Creating list of number of users who authored at least 10 posts in pairs of subreddits: 
#SELECT t1.subreddit, t2.subreddit, SUM(1) as NumOverlaps
#FROM (SELECT subreddit, author, COUNT(1) as cnt
#      FROM [fh-bigquery:reddit_comments.all_starting_201501]
#      WHERE author NOT IN (SELECT author FROM [fh-bigquery:reddit_comments.bots_201505])
#      GROUP BY subreddit, author HAVING cnt > 10) t1
#JOIN (SELECT subreddit, author, COUNT(1) as cnt
#      FROM [fh-bigquery:reddit_comments.all_starting_201501]
#      WHERE author NOT IN (SELECT author FROM [fh-bigquery:reddit_comments.bots_201505])
#      GROUP BY subreddit, author HAVING cnt > 10) t2
#ON t1.author=t2.author
#WHERE t1.subreddit!=t2.subreddit
#GROUP BY t1.subreddit, t2.subreddit

##### Part 1: Load in the data

# CSV files were created by running the SQL code above on Google's BigQuery
shard1=read.table("./data/all_starting_2015_01_overlaps_10com_043017_000000000000.csv",header=TRUE,sep=",",stringsAsFactors=FALSE)
shard2=read.table("./data/all_starting_2015_01_overlaps_10com_043017_000000000001.csv",header=TRUE,sep=",",stringsAsFactors=FALSE)
shard3=read.table("./data/all_starting_2015_01_overlaps_10com_043017_000000000002.csv",header=TRUE,sep=",",stringsAsFactors=FALSE)
allshard = rbind(shard1,shard2,shard3)

##### Part 2: Format and clean data for analysis

# GloVe
sharduniquenames = unique(c(allshard$t1_subreddit,allshard$t2_subreddit))
imatch = match(allshard$t1_subreddit,sharduniquenames)
jmatch = match(allshard$t2_subreddit,sharduniquenames)
shardmat = sparseMatrix(i=imatch,j=jmatch,x=allshard$NumOverlaps,dims=rep(length(sharduniquenames),2),dimnames=list(sharduniquenames,sharduniquenames))
glove = GlobalVectors$new(word_vectors_size=100, vocabulary=sharduniquenames, x_max=100) # GloVe embeddings
glove$fit(x=shardmat, n_iter=25)
subredditvecsglove = glove$get_word_vectors()
# PCA
subredditauthors = colSums(shardmat)
subredditranks = rank(1/subredditauthors)
shardmatforpca = as.matrix(shardmat[,which(subredditranks<=5000)])+1 # Top 5000 subreddits
shardmatforpcasums = rowSums(shardmatforpca)
shardmatforpcanorm = sweep(shardmatforpca,1,shardmatforpcasums,"/")
shardmatforpcasumscontext = colSums(shardmatforpca)
pcacontextprobs = shardmatforpcasumscontext/sum(shardmatforpcasumscontext)
shardmatforpcapmi = log(sweep(shardmatforpcanorm,2,pcacontextprobs,"/")) # PMI version
shardmatforpcappmi = shardmatforpcapmi
shardmatforpcappmi[shardmatforpcapmi<0] = 0
subredditvecspcares = prcomp(shardmatforpcappmi)
subredditvecspca = subredditvecspcares$x[,1:100]
rownames(subredditvecspca) = rownames(shardmat)
# Raw PPMI
subredditvecs = as.matrix(shardmat[,which(subredditranks>200 & subredditranks<=2200)])+1 # Top 2200 without top 200 subreddits
subredditvecssums = apply(subredditvecs,1,sum)
subredditvecsnorm = sweep(subredditvecs,1,subredditvecssums,"/")
subredditvecssumscontext = apply(subredditvecs,2,sum)
contextprobs = subredditvecssumscontext/sum(subredditvecssumscontext)
subredditvecspmi = log(sweep(subredditvecsnorm,2,contextprobs,"/")) # PMI version
subredditvecsppmi = subredditvecspmi
subredditvecsppmi[subredditvecspmi<0] = 0 # PPMI version
scalar1 <- function(x) {x / sqrt(sum(x^2))} # Function to normalize vectors to unit length
subredditvecsppminorm = t(apply(subredditvecsppmi,1,scalar1))
subredditvecspcanorm = t(apply(subredditvecspca,1,scalar1))
subredditvecsglovenorm = t(apply(subredditvecsglove,1,scalar1))

##### Part 3: Evaluation of semantic correctness

## Function to calculate subreddit similarities and perform algebra
# Note that curops always has a leading "+"
findrelsubreddit <- function(cursubs,curops,numret=20,cursubmatuse,cursubmattuse,currownameslcuse) {
    cursubs = tolower(cursubs)
    curvec = 0
    for(i in 1:length(cursubs)) {
	    curvec = ifelse(curops[i]=="+",list(curvec + cursubmatuse[which(currownameslcuse==cursubs[i]),]),list(curvec - cursubmatuse[which(currownameslcuse==cursubs[i]),]))[[1]]
    }
    curclosesubs = cosine(x=curvec,y=cursubmattuse)
    curclosesubso = order(curclosesubs,decreasing=TRUE)
    curclosesubsorder = curclosesubs[curclosesubso]
    curclosesubsorderc = curclosesubsorder[-which(tolower(names(curclosesubsorder))%in%cursubs)]
return(head(curclosesubsorderc,numret))
}

## Automated evaluation of NFL, NHL, and NBA team subreddits
sportssemantics = read.csv(file="./data/subredditautomatedanalogies.csv",header=TRUE,stringsAsFactors=FALSE) # Gold standard relationships
# Function to calculate rank differences
automatedsemanticeval <- function(x,goldstand,cursubmat,cursubmatt,currownameslc) {
	curgoldstand = goldstand[x,]
	cursub1sims = findrelsubreddit(curgoldstand$sub1,c("+"),numret=nrow(cursubmat),cursubmat,cursubmatt,currownameslc)
	cursub2sims = findrelsubreddit(curgoldstand$sub2,c("+"),numret=nrow(cursubmat),cursubmat,cursubmatt,currownameslc)
	curcombsims = findrelsubreddit(c(curgoldstand$sub1,curgoldstand$sub2),c("+",curgoldstand$operation),numret=nrow(cursubmat),cursubmat,cursubmatt,currownameslc)
	cursub1eval = which(tolower(names(cursub1sims))==tolower(curgoldstand$resultsub))
	cursub2eval = which(tolower(names(cursub2sims))==tolower(curgoldstand$resultsub))
	curcombeval = which(tolower(names(curcombsims))==tolower(curgoldstand$resultsub))
	curdiff = min(c(cursub1eval,cursub2eval)) - curcombeval
	curdiffmed = median(c(cursub1eval,cursub2eval)) - curcombeval
returndata = list(sub1=cursub1eval,sub2=cursub2eval,comb=curcombeval,diff=curdiff,diffmed=curdiffmed)
return(returndata)
}

# Calculate rank differences for each method
cursubmatset = subredditvecsppminorm # PPMI matrix
cursubmattset = t(cursubmatset)
currownameslcset = tolower(rownames(cursubmatset))
ppmirankeval = mclapply(1:nrow(sportssemantics),automatedsemanticeval,sportssemantics,cursubmatset,cursubmattset,currownameslcset)
ppmirankevaldf = data.frame(sportssemantics,sub1=unlist(sapply(ppmirankeval,"[","sub1")),sub2=unlist(sapply(ppmirankeval,"[","sub2")),comb=unlist(sapply(ppmirankeval,"[","comb")),diff=unlist(sapply(ppmirankeval,"[","diff")),diffmed=unlist(sapply(ppmirankeval,"[","diffmed")))
cursubmatset = subredditvecspcanorm # PCA matrix
cursubmattset = t(cursubmatset)
currownameslcset = tolower(rownames(cursubmatset))
pcarankeval = mclapply(1:nrow(sportssemantics),automatedsemanticeval,sportssemantics,cursubmatset,cursubmattset,currownameslcset)
pcarankevaldf = data.frame(sportssemantics,sub1=unlist(sapply(pcarankeval,"[","sub1")),sub2=unlist(sapply(pcarankeval,"[","sub2")),comb=unlist(sapply(pcarankeval,"[","comb")),diff=unlist(sapply(pcarankeval,"[","diff")),diffmed=unlist(sapply(pcarankeval,"[","diffmed")))
cursubmatset = subredditvecsglovenorm # GloVe matrix
cursubmattset = t(cursubmatset)
currownameslcset = tolower(rownames(cursubmatset))
gloverankeval = mclapply(1:nrow(sportssemantics),automatedsemanticeval,sportssemantics,cursubmatset,cursubmattset,currownameslcset)
gloverankevaldf = data.frame(sportssemantics,sub1=unlist(sapply(gloverankeval,"[","sub1")),sub2=unlist(sapply(gloverankeval,"[","sub2")),comb=unlist(sapply(gloverankeval,"[","comb")),diff=unlist(sapply(gloverankeval,"[","diff")),diffmed=unlist(sapply(gloverankeval,"[","diffmed")))

# Calculate median and wilcoxon CI for each method
ppmimedianCIh = aggregate(ppmirankevaldf[c("diffmed")],by=list(ppmirankevaldf$sub1),FUN=wilcox.test,conf.int=TRUE,simplify=FALSE)
ppmimedianh = aggregate(ppmirankevaldf[,c("comb")],by=list(ppmirankevaldf$sub1),FUN=median)
ppmimedianCI = data.frame(method="Explicit",league=toupper(ppmimedianCIh$Group.1),mediancomb=ppmimedianh$x,mediandiff=unlist(sapply(ppmimedianCIh$diff,"[","estimate")),CIlow=sapply(sapply(ppmimedianCIh$diff,"[","conf.int"),"[",1),CIhigh=sapply(sapply(ppmimedianCIh$diff,"[","conf.int"),"[",2),pvalue=unlist(sapply(ppmimedianCIh$diff,"[","p.value")),row.names=paste0("ppmi",1:3))
pcamedianCIh = aggregate(pcarankevaldf[c("diffmed")],by=list(pcarankevaldf$sub1),FUN=wilcox.test,conf.int=TRUE,simplify=FALSE)
pcamedianh = aggregate(pcarankevaldf[,c("comb")],by=list(pcarankevaldf$sub1),FUN=median)
pcamedianCI = data.frame(method="PCA",league=toupper(pcamedianCIh$Group.1),mediancomb=pcamedianh$x,mediandiff=unlist(sapply(pcamedianCIh$diff,"[","estimate")),CIlow=sapply(sapply(pcamedianCIh$diff,"[","conf.int"),"[",1),CIhigh=sapply(sapply(pcamedianCIh$diff,"[","conf.int"),"[",2),pvalue=unlist(sapply(pcamedianCIh$diff,"[","p.value")),row.names=paste0("pca",1:3))
glovemedianCIh = aggregate(gloverankevaldf[c("diffmed")],by=list(gloverankevaldf$sub1),FUN=wilcox.test,conf.int=TRUE,simplify=FALSE)
glovemedianh = aggregate(gloverankevaldf[,c("comb")],by=list(gloverankevaldf$sub1),FUN=median)
glovemedianCI = data.frame(method="GloVe",league=toupper(glovemedianCIh$Group.1),mediancomb=glovemedianh$x,mediandiff=unlist(sapply(glovemedianCIh$diff,"[","estimate")),CIlow=sapply(sapply(glovemedianCIh$diff,"[","conf.int"),"[",1),CIhigh=sapply(sapply(glovemedianCIh$diff,"[","conf.int"),"[",2),pvalue=unlist(sapply(glovemedianCIh$diff,"[","p.value")),row.names=paste0("glove",1:3))

# Plot results of each method stratified by sport type
allmethodmedianCI = rbind(ppmimedianCI,pcamedianCI,glovemedianCI)
png("./plots/allrankeval.png",width=7,height=3,res=350,units="in")
ggplot(data=allmethodmedianCI, aes(x=method, y=mediandiff, fill=method)) + geom_bar(position=position_dodge(), stat="identity") + geom_errorbar(aes(ymin=CIlow, ymax=CIhigh), width=.2, position=position_dodge(.9)) + facet_grid(.~league) + theme_bw() + xlab("Method") + ylab("Median Rank Diff.") + theme(legend.title=element_blank())
dev.off()

## Looking at some interesting selected examples

cursubmatset = subredditvecsppminorm
cursubmattset = t(cursubmatset)
currownameslcset = tolower(rownames(cursubmatset))
findrelsubreddit(c("personalfinance","frugal"),c("+","-"),5,cursubmatset,cursubmattset,currownameslcset)
findrelsubreddit(c("chicagobulls","chicago","minnesota"),c("+","-","+"),5,cursubmatset,cursubmattset,currownameslcset)
findrelsubreddit(c("berlin","germany","unitedkingdom"),c("+","-","+"),5,cursubmatset,cursubmattset,currownameslcset)
findrelsubreddit(c("running","weightlifting"),c("+","+"),5,cursubmatset,cursubmattset,currownameslcset)
findrelsubreddit(c("books","fiction"),c("+","+"),5,cursubmatset,cursubmattset,currownameslcset)

##### Part 4: tSNE Clustering

set.seed(123) # For reproducibility
curtsnemat = subredditvecsglovenorm[order(subredditvecssums,decreasing=TRUE),][1:5000,]
rtsne_out = Rtsne(curtsnemat,dims=2,check_duplicates=FALSE,perplexity=50,verbose=TRUE,pca=FALSE)
png("./plots/tsneplotnames.png", width=20, height=20,units="in",res=350)
plot(rtsne_out$Y, t='n', main="BarnesHutSNE")
text(rtsne_out$Y, labels=rownames(curtsnemat),cex=.5)
dev.off()
tsneforggplot = data.frame(x=rtsne_out$Y[,1],y=rtsne_out$Y[,2])
rownames(tsneforggplot) = rownames(curtsnemat)
tsneinset = ggplot(data=tsneforggplot,aes(x=x,y=y)) + geom_point(alpha=.7,size=.05) + theme_bw() + theme(panel.grid.major=element_blank(),panel.grid.minor=element_blank(),axis.title=element_blank(),axis.ticks=element_blank(),axis.text=element_text(size=7),text=element_text(size=8)) + ggtitle("Full tSNE Clustering")
medicalregion = c(-27,-17,-20,-15)
musicregion = c(3,15,10,18)
# Medical region
curregion = medicalregion
tsneforggplotmedical = subset(tsneforggplot,x>=curregion[1] & x<=curregion[2] & y>=curregion[3] & y<=curregion[4])
tsnemedical = ggplot(data=tsneforggplotmedical,aes(x=x,y=y)) + geom_point(alpha=.7) + theme_bw() + theme(panel.grid.major=element_blank(),panel.grid.minor=element_blank(),axis.title=element_blank()) + geom_label_repel(aes(label=rownames(tsneforggplotmedical))) + ggtitle("Medical Cluster")
tsneinsetmedical = tsneinset + geom_rect(aes(xmin=curregion[1],xmax=curregion[2],ymin=curregion[3],ymax=curregion[4]),fill=alpha("red",.05))
vpmedical = viewport(width = 0.25, height = 0.25, x = .2, y = .25)
png("./plots/tsnemedical.png", width=7, height=7,units="in",res=350)
print(tsnemedical)
print(tsneinsetmedical,vp=vpmedical)
dev.off()
# Music region
curregion = musicregion
tsneforggplotmusicdownsample = tsneforggplot[1:2500,]
tsneforggplotmusic = subset(tsneforggplotmusicdownsample,x>=curregion[1] & x<=curregion[2] & y>=curregion[3] & y<=curregion[4])
tsnemusic = ggplot(data=tsneforggplotmusic,aes(x=x,y=y)) + geom_point(alpha=.7) + theme_bw() + theme(panel.grid.major=element_blank(),panel.grid.minor=element_blank(),axis.title=element_blank()) + geom_label_repel(aes(label=rownames(tsneforggplotmusic))) + ggtitle("Music Cluster")
tsneinsetmusic = tsneinset + geom_rect(aes(xmin=curregion[1],xmax=curregion[2],ymin=curregion[3],ymax=curregion[4]),fill=alpha("red",.05))
vpmusic = viewport(width = 0.25, height = 0.25, x = .85, y = .72)
png("./plots/tsnemusic.png", width=7, height=7,units="in",res=350)
print(tsnemusic)
print(tsneinsetmusic,vp=vpmusic)
dev.off()


