plugins { id("io.vacco.common-build") version "0.5.3" }

group = "io.vacco.jtinn"
version = "1.0.1"

configure<io.vacco.common.CbPluginProfileExtension> {
  addJ8Spec()
  addPmd()
  addSpotBugs()
  addClasspathHell()
  setPublishingUrlTransform { repo -> "${repo.url}/${project.name}" }
  sharedLibrary()
}

configure<JavaPluginExtension> {
  sourceCompatibility = JavaVersion.VERSION_1_8
  targetCompatibility = JavaVersion.VERSION_1_8
}
