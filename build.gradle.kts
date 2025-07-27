plugins { id("io.vacco.oss.gitflow") version "1.8.2" }

group = "io.vacco.jtinn"
version = "3.8.0"

configure<io.vacco.oss.gitflow.GsPluginProfileExtension> {
  addJ8Spec()
  sharedLibrary(true, false)
}

tasks.named<ProcessResources>("processResources") {
  from("src/main/c/libjtinn.so")    { into("io/vacco/jtinn") }
  from("src/main/c/libjtinn.dylib") { into("io/vacco/jtinn") }
}

tasks.jacocoTestReport {
  reports {
    xml.required = true
    csv.required = true
  }
}
