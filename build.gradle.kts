plugins { id("io.vacco.oss.gitflow") version "1.8.2" }

group = "io.vacco.jtinn"
version = "3.8.0"

configure<io.vacco.oss.gitflow.GsPluginProfileExtension> {
  addJ8Spec()
  sharedLibrary(true, false)
}

dependencies {
  testImplementation("com.google.code.gson:gson:2.11.0")
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

tasks.test {
  maxHeapSize = "2048g"
  jvmArgs("-XX:StartFlightRecording=duration=300s,filename=build/test-recording.jfr")
}

tasks.register("dumpJfrFiltered") {
  dependsOn("test") // Ensures test task runs first to generate the .jfr file
  doLast {
    val javaHome = System.getProperty("java.home")
    val osName = System.getProperty("os.name").lowercase()
    val jfrExecutable = if (osName.contains("win")) "jfr.exe" else "jfr"
    val jfrPath = "$javaHome/bin/$jfrExecutable"
    val jfrFile = file("build/test-recording.jfr")
    val outputFile = file("build/jfr_filtered_execution_samples.json")
    // Run jfr print with JSON output, filtering for jdk.ExecutionSample events
    exec {
      commandLine(jfrPath, "print", "--json", "--events", "jdk.ExecutionSample", jfrFile.absolutePath)
      standardOutput = outputFile.outputStream()
    }
    println("Filtered JFR data (ExecutionSample events) dumped to ${outputFile.absolutePath}")
  }
  mustRunAfter("test")
}
