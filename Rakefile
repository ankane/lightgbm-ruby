require "bundler/gem_tasks"
require "rake/testtask"

task default: :test
Rake::TestTask.new do |t|
  t.libs << "test"
  t.pattern = "test/**/*_test.rb"
  t.warning = false # for daru
end

shared_libraries = %w(lib_lightgbm.dll lib_lightgbm.dylib lib_lightgbm.arm64.dylib lib_lightgbm.so)

# ensure vendor files exist
task :ensure_vendor do
  shared_libraries.each do |file|
    raise "Missing file: #{file}" unless File.exist?("vendor/#{file}")
  end
end

Rake::Task["build"].enhance [:ensure_vendor]

def download_file(file, sha256)
  require "open-uri"

  # also update licenses in vendor/
  version = "3.3.5"

  url =
    if file == "lib_lightgbm.arm64.dylib"
      "https://github.com/ankane/ml-builds/releases/download/lightgbm-#{version}/#{file}"
    else
      "https://github.com/microsoft/LightGBM/releases/download/v#{version}/#{file}"
    end
  puts "Downloading #{file}..."
  contents = URI.open(url).read

  computed_sha256 = Digest::SHA256.hexdigest(contents)
  raise "Bad hash: #{computed_sha256}" if computed_sha256 != sha256

  dest = "vendor/#{file}"
  File.binwrite(dest, contents)
  puts "Saved #{dest}"
end

# https://github.com/microsoft/LightGBM/releases
namespace :vendor do
  task :linux do
    download_file("lib_lightgbm.so", "643f120154a5f72f50add9776f606db1e8ad401f975d8495552d15e1c8d72515")
  end

  task :mac do
    download_file("lib_lightgbm.dylib", "6eb9ddf9d45d4af9f7a33692e572d9eaac4f3e4dbcf3a9aba9251cf4d0918122")
    download_file("lib_lightgbm.arm64.dylib", "eb6e809b22ef7c09dcfcfda028b3fd229a6e409437df25a741c69021d224d5af")
  end

  task :windows do
    download_file("lib_lightgbm.dll", "41dcd97e02e2af695ebc9bf791c77a40a7e6961ef357fea91ca16ab021509013")
  end

  task all: [:linux, :mac, :windows]

  task :platform do
    if Gem.win_platform?
      Rake::Task["vendor:windows"].invoke
    elsif RbConfig::CONFIG["host_os"] =~ /darwin/i
      Rake::Task["vendor:mac"].invoke
    else
      Rake::Task["vendor:linux"].invoke
    end
  end
end
