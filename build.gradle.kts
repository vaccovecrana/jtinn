plugins {
  `java-library`
  `maven-publish`
}

group = "io.vacco.jtinn"
version = "1.0"

java {
  withJavadocJar()
  withSourcesJar()
  sourceCompatibility = org.gradle.api.JavaVersion.VERSION_1_8
  targetCompatibility = org.gradle.api.JavaVersion.VERSION_1_8
}

repositories { jcenter() }

dependencies {
  testImplementation("io.github.j8spec:j8spec:3.0.0")
}

publishing {
  publications {
    create<MavenPublication>("mavenJava") {
      from(components["java"])
      versionMapping {
        usage("java-api") { fromResolutionOf("runtimeClasspath") }
        usage("java-runtime") { fromResolutionResult() }
      }
    }
  }
  repositories {
    maven {
      name = "bintray"
      val bintrayUser = System.getenv("BINTRAY_USER")
      setUrl("https://api.bintray.com/maven/vaccovecrana/${project.group}/${project.name}/;publish=0;override=1")
      credentials {
        username = bintrayUser
        password = System.getenv("BINTRAY_KEY")
      }
    }
  }
}
