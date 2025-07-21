plugins { id("io.vacco.oss.gitflow") version "1.8.0" }

group = "io.vacco.jtinn"
version = "3.0.1"

configure<io.vacco.oss.gitflow.GsPluginProfileExtension> {
  addJ8Spec()
  addClasspathHell()
  sharedLibrary(true, false)
}
