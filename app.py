import os
import secrets
import logging
import jwt
from typing import List, Dict
from datetime import timedelta, datetime
from passlib.context import CryptContext
from fastapi import FastAPI, Form, HTTPException, Request, Depends
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.security import OAuth2PasswordBearer
from pydantic import BaseModel
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from pinecone import Pinecone
from langchain_pinecone import PineconeVectorStore
from urllib.parse import quote, unquote

# 환경 변수 및 상수
load_dotenv()
SECRET_KEY = os.getenv("JWT_SECRET_KEY")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

# Pinecone 초기화
pc = Pinecone(api_key="PINECONE_API_KEY")
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
vector_store = PineconeVectorStore.from_existing_index("recipes", embeddings)

# FastAPI 앱 초기화
app = FastAPI(
    title="CheftGPT. The best provider of Recipes in the world.",
    description="Give ChefGPT the name of an ingredient and it will give you multiple recipes to use that ingredient on in return.",
    servers=[{"url": "https://realistic-filed-donors-trailer.trycloudflare.com"}],
)

# 데이터베이스 초기화
users_db: Dict[str, Dict] = {}
token_db: Dict[str, str] = {}
access_tokens_db: Dict[str, Dict] = {}

# 인증 관련 설정
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")


# Pydantic 모델 정의
class Document(BaseModel):
    page_content: str


class Recipe(BaseModel):
    name: str
    ingredients: List[str]
    instructions: str
    link: str


class FavoriteResponse(BaseModel):
    status: str
    message: str
    recipe: Recipe


# 인증 관련 함수
def create_access_token(data: dict, expires_delta: timedelta = None):
    to_encode = data.copy()
    expire = datetime.now() + (
        expires_delta or timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    )
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)


def decode_token_to_get_username(token: str):
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username = payload.get("sub")
        if username is None:
            raise HTTPException(
                status_code=401, detail="Token does not contain username"
            )
        return username
    except jwt.PyJWTError:
        raise HTTPException(status_code=401, detail="Could not validate credentials")


def get_current_username(token: str = Depends(oauth2_scheme)):
    if not token:
        logging.error("Authorization header is missing")
        raise HTTPException(status_code=401, detail="Authorization header missing")
    username = decode_token_to_get_username(token)
    if not username:
        logging.error("Invalid token provided")
        raise HTTPException(status_code=401, detail="Invalid token")
    return username


# API
@app.get("/authorize", response_class=HTMLResponse, include_in_schema=False)
def handle_authorize(client_id: str, redirect_uri: str, state: str):
    return f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <title>Log In or Register</title>
    </head>
    <body>
        <h1>Log In or Register</h1>
        <form action="/authorize" method="post">
            <input type="hidden" name="redirect_uri" value="{redirect_uri}">
            <input type="hidden" name="state" value="{state}">
            <label for="username">Username:</label>
            <input type="text" id="username" name="username" required><br><br>
            <label for="password">Password:</label>
            <input type="password" id="password" name="password" required><br><br>
            <button type="submit" name="action" value="login">Log In</button>
            <button type="submit" name="action" value="register">Register</button>
        </form>
    </body>
    </html>
    """


@app.post("/authorize", response_class=HTMLResponse, include_in_schema=False)
async def authorize_user(
    request: Request,
    username: str = Form(...),
    password: str = Form(...),
    action: str = Form(...),
    redirect_uri: str = Form(...),
    state: str = Form(...),
):
    if action == "register":
        if username in users_db:
            return HTMLResponse("Username already registered", status_code=400)
        users_db[username] = {
            "username": username,
            "password": pwd_context.hash(password),
        }
    elif action == "login":
        user = users_db.get(username)
        if not user or not pwd_context.verify(password, user["password"]):
            return HTMLResponse("Incorrect username or password", status_code=401)

    code = secrets.token_urlsafe(16)
    token_db[code] = username
    decoded_redirect = unquote(redirect_uri)
    return RedirectResponse(
        url=f"{decoded_redirect}?code={code}&state={state}", status_code=303
    )


@app.post("/token", include_in_schema=False)
async def login_for_access_token(code: str = Form(...)):
    username = token_db.get(code)
    if not username:
        raise HTTPException(status_code=400, detail="Invalid code")

    token_info = access_tokens_db.get(code)
    if token_info and token_info.get("expires") > datetime.now():
        return {"access_token": token_info["access_token"], "token_type": "bearer"}

    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": username}, expires_delta=access_token_expires
    )

    access_tokens_db[code] = {
        "access_token": access_token,
        "expires": datetime.now() + access_token_expires,
    }
    return {"access_token": access_token, "token_type": "bearer"}


@app.get("/recipes", response_model=list[Document])
async def get_recipe(ingredient: str, username: str = Depends(get_current_username)):
    if not username:
        raise HTTPException(detail="Authentication required")
    return vector_store.similarity_search(ingredient)


@app.post("/add_favorite", response_model=FavoriteResponse)
async def add_favorite_recipe(
    recipe: Recipe, username: str = Depends(get_current_username)
):
    if username not in users_db:
        raise HTTPException(status_code=404, detail="User not found")
    if "favorite_recipes" not in users_db[username]:
        users_db[username]["favorite_recipes"] = []
    users_db[username]["favorite_recipes"].append(recipe.dict())
    return {
        "status": "success",
        "message": "Recipe added to favorites",
        "recipe": recipe,
    }


@app.get("/favorites", response_model=List[Recipe])
async def display_favorite_recipes(
    request: Request, username: str = Depends(get_current_username)
):
    if username not in users_db or "favorite_recipes" not in users_db[username]:
        return HTMLResponse(
            "<html><body><h1>No favorite recipes found.</h1></body></html>"
        )

    favorite_recipes = users_db[username]["favorite_recipes"]
    if not favorite_recipes:
        return HTMLResponse(
            "<html><body><h1>No favorite recipes found.</h1></body></html>"
        )

    html_content = "<html><head><title>Favorite Recipes</title></head><body>"
    html_content += "<h1>Favorite Recipes</h1>"
    for recipe in favorite_recipes:
        html_content += f"""
            <h2>{recipe['name']}</h2>
            <h3>Ingredients:</h3><ul>
            {''.join(f"<li>{ingredient}</li>" for ingredient in recipe['ingredients'])}
            </ul>
            <h3>Instructions:</h3><p>{recipe['instructions']}</p>
            <h3>Link:</h3><p>{recipe['link']}</p>
        """
    html_content += "</body></html>"
    return HTMLResponse(content=html_content)
