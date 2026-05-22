import subprocess
import sys
import os

def generate_missing_requirements():
    """
    比较旧的 requirements.txt 和当前环境，生成一个只包含缺失包的新文件。
    """
    # 旧的依赖文件路径
    old_req_file = 'requirements.txt'
    # 将要生成的新文件路径
    missing_req_file = 'missing_requirements.txt'

    if not os.path.exists(old_req_file):
        print(f"❌ 错误: 未在当前目录找到 '{old_req_file}' 文件。")
        return

    print("🔍 正在获取当前环境已安装的包列表...")
    try:
        # 使用当前环境的 Python 解释器来运行 pip freeze
        result = subprocess.run(
            [sys.executable, '-m', 'pip', 'freeze'],
            capture_output=True,
            text=True,
            check=True,
            encoding='utf-8'
        )
        installed_packages_raw = result.stdout.splitlines()
    except subprocess.CalledProcessError as e:
        print(f"❌ 获取已安装包列表时出错: {e}")
        return

    # 将已安装的包名存入一个 set 中，用于快速查找（忽略版本号，只关心包名）
    installed_package_names = set()
    for pkg in installed_packages_raw:
        if '==' in pkg:
            # 转为小写以忽略大小写差异
            installed_package_names.add(pkg.split('==')[0].lower())

    print(f"📖 正在读取 '{old_req_file}'...")
    with open(old_req_file, 'r', encoding='utf-8') as f:
        old_reqs = f.read().splitlines()

    # 找出缺失的包
    missing_reqs = []
    print("⚖️ 正在比较差异...")
    for req in old_reqs:
        req = req.strip()
        if req and not req.startswith('#') and '==' in req:
            name = req.split('==')[0].lower()
            # 如果旧依赖中的包名不在新环境的已安装列表中，则判定为缺失
            if name not in installed_package_names:
                missing_reqs.append(req)

    # 将缺失的包写入新文件
    if missing_reqs:
        print(f"✅ 发现 {len(missing_reqs)} 个缺失的包。正在生成 '{missing_req_file}'...")
        with open(missing_req_file, 'w', encoding='utf-8') as f:
            for req in missing_reqs:
                f.write(req + '\n')
        print("\n🎉 操作成功！")
        print(f"现在请在你的新环境 (speedXY_py310) 中运行以下命令来安全地安装这些包:")
        print(f"pip install -r {missing_req_file}")
    else:
        print("✅ 检查完成，没有发现任何缺失的包。你的环境看起来已经满足要求。")

if __name__ == '__main__':
    generate_missing_requirements()
