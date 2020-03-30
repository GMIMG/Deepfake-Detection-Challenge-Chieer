FROM openjdk:13-jdk

# COPY ./application /application
# WORKDIR /application
# CMD ["./gradlew", "bootRun"]

ADD app.jar app.jar
CMD ["java", "-jar", "app.jar"]