![Git 브랜칭 기법](https://t1.daumcdn.net/cfile/tistory/99DEE0415AB07EF817)



내 폴더로 가져오기 (한번만 하면됨)

```bash
git clone https://github.com/GMIMG/Deepfake-Detection-Challenge-Chieer.git
```



브랜치

```bash
# 브랜치 확인
git branch
# 브랜치 생성
git branch "branch_name"
# 브랜치 전환
git checkout "branch_name"
```



각자의 브랜치에 커밋 & 푸쉬

```bash
# 커밋 & 푸쉬
git add .
git commit -m "commit_message"
# 업로드
git push origin "branch_name"
```



합치고 싶을때

```bash
# 최신상태 갱신
# git pull origin "develop"

# 브랜치 전환
git checkout "develop"
# develop으로 merge
git merge --no-ff "branch_name"
git push origin develop

# 전체 branch 보기
git branch -a
# tree 확인
gitk
```



참고

[우아한 형제들 기술블로그](https://woowabros.github.io/experience/2017/10/30/baemin-mobile-git-branch-strategy.html)