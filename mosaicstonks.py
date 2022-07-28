#!/usr/bin/env python
# coding: utf-8

# In[2]:


import praw
import pandas as pd
import os
import requests
import shutil
import glob
from scipy import spatial
import numpy as np
from PIL import Image
from datetime import datetime
import time


def GetConfig():
    with open("config.json") as f:
        config = json.load(f)["config"]
        client_id = config["client_id"]
        client_secret = config["client_secret"]
        password = config["password"]
        user_agent = config["user_agent"]
        username = config["username"]
        img_path = config["img_path"]
        img_template_path = config["img_template_path"]
        img_results_path = config["img_results_path"]

    return client_id, client_secret, password, user_agent, username, img_path, img_template_path, img_results_path


def GetMode():
    while True:
        images_downloaded = input(
            "Type 'Y' if you already downloaded images, type 'N' if you want to crawl new posts"
        )
        if images_downloaded.lower() == "y" or images_downloaded.lower(
        ) == "n":
            break
    return images_downloaded


def GetSubreddits():
    while True:
        subreddits = input(
            'Please enter a list of subreddits, separated by "comma". For example: "Superstonk, GME".'
        )
        subreddits = [subreddit.strip() for subreddit in subreddits.split(",")]
        [subreddit for subreddit in subreddits if subreddit != ""]
        if len(subreddits) != 0:
            break

    return subreddits


def DefineLimit():
    while True:
        limit = input('Please enter a limit of posts. For example: "500".')
        try:
            int(limit)
            break
        except:
            pass

    return int(limit)


def CreateFolderStructure(img_path, img_template_path, img_results_path):
    try:
        os.mkdir(img_path)
    except:
        pass

    try:
        os.mkdir(img_template_path)
    except:
        pass

    try:
        os.mkdir(img_results_path)
    except:
        pass
    return img_path, img_template_path, img_results_path


def SaveImageToDisk(url, img_path):
    r = requests.get(url, stream=True)
    if r.status_code == 200:
        with open(f'{img_path}/{url.split("/")[-1]}', 'wb') as f:
            r.raw.decode_content = True
            shutil.copyfileobj(r.raw, f)


def GetSubmissionsAll(reddit, subreddits, limit, only_media):
    df_posts = pd.DataFrame()
    df_list = []
    for subreddit in subreddits:
        print(f'\nSubreddit: {subreddit}')
        counter = 0
        for post in reddit.subreddit(subreddit).hot(limit=limit):
            counter += 1
            print(f'Post: {counter} {post}')
            try:
                df_posts_temp = pd.DataFrame()
                df_posts_temp.at[post, "post_id"] = post.id
                df_posts_temp.at[post, "title"] = post.title
                df_posts_temp.at[post, "author"] = post.author.name
                df_posts_temp.at[post, "flair"] = post.link_flair_text
                df_posts_temp.at[post, "subreddit"] = post.subreddit
            except:
                pass

            try:
                df_posts_temp.at[post, "media"] = post.is_reddit_media_domain
                df_posts_temp.at[post, "image_url"] = post.url
            except:
                df_posts_temp.at[post, "media"] = None
                df_posts_temp.at[post, "image_url"] = None
            df_list.append(df_posts_temp)
            df_posts = pd.concat(df_list)

    if only_media == True:
        #filter to media and non-CS posts
        df_posts = df_posts[df_posts["media"] == True].reset_index(drop=True)

    return df_posts


def DownloadImages(img_path, df_posts):
    # download images
    for row in df_posts.index:
        print(f'Downloading image {row+1}/{len(df_posts)}')
        if ".gif" not in df_posts.at[row, "image_url"]:
            SaveImageToDisk(df_posts.at[row, "image_url"], img_path)
    return "Images downloaded."


def DeleteCorruptImages(img_path):
    # delete corrupt and too big images
    [
        os.remove(f"{img_path}/{img}") for img in os.listdir(img_path)
        if "." not in img or os.path.getsize(f"{img_path}/{img}") >= 7000000
    ]
    return "Images cleaned."


def ConvertImagesToRGB(img_path):
    # convert images to rgb
    counter = 0
    for image in os.listdir(img_path):
        counter += 1
        if "." in image:
            print(f'Converting image {counter}/{len(os.listdir(img_path))}')
            Image.open(f'{img_path}/{image}').convert("RGB").save(
                f'{img_path}/{image}')
    return "Images converted."


def MakeMosaicImage(template_image, img_template_path, img_path,
                    img_results_path, tile_size):

    # Sources and settings
    tile_photos_path = f"{img_path}\\*"

    # Get all tiles
    tile_paths = []
    for file in glob.glob(tile_photos_path):
        tile_paths.append(file)

    # Import and resize all tiles
    tiles = []
    for path in tile_paths:
        tile = Image.open(path)
        tile = tile.resize((tile_size, tile_size))
        tiles.append(tile)

    # Calculate dominant color
    colors = []
    for tile in tiles:
        mean_color = np.array(tile).mean(axis=0).mean(axis=0)
        colors.append(mean_color)

    # Pixelate (resize) main photo
    main_photo = Image.open(f'{img_template_path}/{template_image}')

    width = int(np.round(main_photo.size[0] / tile_size))
    height = int(np.round(main_photo.size[1] / tile_size))
    resized_photo = main_photo.resize((width, height))

    # Find closest tile photo for every pixel

    # Create a KDTree
    tree = spatial.KDTree(colors)

    # Empty integer array to store indices of tiles
    closest_tiles = np.zeros((width, height), dtype=np.uint32)

    for i in range(width):
        for j in range(height):
            pixel = resized_photo.getpixel(
                (i, j))  # Get the pixel color at (i, j)
            closest = tree.query(pixel)  # Returns (distance, index)
            closest_tiles[i, j] = closest[1]  # We only need the index

    # Create an output image
    output = Image.new('RGB', main_photo.size)

    # Draw tiles
    for i in range(width):
        for j in range(height):
            # Offset of tile
            x, y = i * tile_size, j * tile_size
            # Index of tile
            index = closest_tiles[i, j]
            # Draw tile
            output.paste(tiles[index], (x, y))

    # Save output
    timestamp = datetime.now().strftime("%H%M%S")

    output.save(
        f"{img_results_path}/{template_image.split('.')[0]}_{timestamp}_{tile_size}x{tile_size}.jpg"
    )
    return "Image file generated."


def main():
    # get config
    client_id, client_secret, password, user_agent, username, img_path, img_template_path, img_results_path = GetConfig(
    )

    # define mode
    images_downloaded = GetMode()

    if images_downloaded == "n":
        # define subreddits
        subreddits = GetSubreddits()

        # define limit
        limit = DefineLimit()

        # create base structure
        CreateFolderStructure(img_path, img_template_path, img_results_path)

        # initiate reddit session
        reddit = praw.Reddit(client_id=client_id,
                             client_secret=client_secret,
                             password=password,
                             user_agent=user_agent,
                             username=username)

        # crawl reddit (all posts)
        df_posts = GetSubmissionsAll(reddit, subreddits, limit, True)

        # exclude purple circles
        df_posts = df_posts[
            df_posts["flair"] != str("💻 Computershare")].reset_index(drop=True)

        # download images
        DownloadImages(img_path, df_posts)

        # clean images
        DeleteCorruptImages(img_path)

        # convert images
        ConvertImagesToRGB(img_path)

    # run process
    for template_image in os.listdir(img_template_path):
        print(f'Creating "{template_image.split(".")[0]}" image...')
        MakeMosaicImage(template_image,
                        img_template_path,
                        img_path,
                        img_results_path,
                        tile_size=40) #change tile size if you want (the lower the more detailled)


if __name__ == "__main__":
    input(
        "Please make sure that you have images in the folder 'images_templates' before you proceed. These images will serve as templates.\nPlease press any key to proceed.\n"
    )
    main()

