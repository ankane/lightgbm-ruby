require "bundler/gem_tasks"
require "rake/testtask"

Rake::TestTask.new do |t|
  t.pattern = "test/**/*_test.rb"
  t.warning = false # for daru
end

task default: :test

shared_libraries = %w(lib_lightgbm.dll lib_lightgbm.dylib lib_lightgbm.arm64.dylib lib_lightgbm.so lib_lightgbm.arm64.so)

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
  version = "4.6.0"

  url =
    if ["lib_lightgbm.arm64.so", "lib_lightgbm.arm64.dylib"].include?(file)
      "https://github.com/ankane/ml-builds/releases/download/lightgbm-#{version}/#{file}"
    else
      "https://github.com/microsoft/LightGBM/releases/download/v#{version}/#{file}"
    end
  puts "Downloading #{file}..."
  contents = URI.parse(url).read

  computed_sha256 = Digest::SHA256.hexdigest(contents)
  raise "Bad hash: #{computed_sha256}" if computed_sha256 != sha256

  dest = "vendor/#{file}"
  File.binwrite(dest, contents)
  puts "Saved #{dest}"
end

# https://github.com/microsoft/LightGBM/releases
namespace :vendor do
  task :linux do
    download_file("lib_lightgbm.so", "237f15e1362a5abab4be0fae14aebba7bb278763f3412a82c50ab9d1fc0dc8bd")
    download_file("lib_lightgbm.arm64.so", "748e78afbb275d03b0de114d9d629e3b31e397f052c3fe9fe7d4e2fddc7e536e")
  end

  task :mac do
    download_file("lib_lightgbm.dylib", "15c6678c60f1acf4a34f0784f799ee3ec7a48e25efa9be90e7415d54f9bed858")
    download_file("lib_lightgbm.arm64.dylib", "df56dce6597389a749de75e46b5383f83c751f57da643232ef766f15aca10a0d")
  end

  task :windows do
    download_file("lib_lightgbm.dll", "a5032c5278f3350ea9f7925b7b4d270b23af9a8e9639971cb025d615b45c39e7")
  end

  task all: [:linux, :mac, :windows]

  task :platform do
    if Gem.win_platform?
      Rake::Task["vendor:windows"].invoke
    elsif RbConfig::CONFIG["host_os"].match?(/darwin/i)
      Rake::Task["vendor:mac"].invoke
    else
      Rake::Task["vendor:linux"].invoke
    end
  end
end
